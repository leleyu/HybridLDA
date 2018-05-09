package lda.parallel;

import lda.utils.DVMat;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Created by leleyu on 2016/9/29.
 */
public class WarpLDA {

  public int D, V, K;
  public float alpha, beta, vbeta;
  public int MH_STEPS;

  public DVMat mat;
  public int[] nk;

  public int maxlen;

  public double lgamma_beta;
  public double lgamma_alpha;


  Task[] rowTasks;
  Task[] colTasks;
  Task[] initTasks;
  Future<Double>[] futures;

  public int taskNum;
  public ThreadPoolExecutor executor;

  public boolean test = false;

  public WarpLDA(int D, int V, int K, float alpha, float beta,
                 DVMat mat, int taskNum) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta = beta;
    this.vbeta = V * beta;
    this.mat = mat;
    this.MH_STEPS = mat.MH_STEPS;

    this.lgamma_alpha = Gamma.logGamma(alpha);
    this.lgamma_beta = Gamma.logGamma(beta);

    this.taskNum = taskNum;
    this.nk = new int[K];

    initTasks();

    maxlen = 0;
    for (int d = 0; d < D; d ++) {
      int len = mat.ds[d + 1] - mat.ds[d];
      maxlen = Math.max(len, maxlen);
    }
  }

  public void initTasks() {
    this.rowTasks = new RowTask[taskNum];
    this.colTasks = new ColTask[taskNum];
    this.initTasks = new InitTask[taskNum];
    this.futures = new Future[taskNum];
    for (int i = 0; i < taskNum; i++) {
      rowTasks[i] = new RowTask();
      colTasks[i] = new ColTask();
      initTasks[i] = new InitTask();
    }

    executor = new ThreadPoolExecutor(16, 32, 60,
            TimeUnit.HOURS,
            new LinkedBlockingQueue<Runnable>());
  }

  public void init(int w, Random rand, int[] ws, short[] mhs, short[] topics, int[] nk) {
    for (int wi = ws[w]; wi < ws[w + 1]; wi++) {
      short tt = (short) rand.nextInt(K);
      topics[wi] = tt;
      nk[tt]++;
      for (int m = 0; m < MH_STEPS; m++) {
        mhs[wi * MH_STEPS + m] = tt;
      }
    }
  }

  public double visitByRow(int d, Random rand, short[] topics, short[] mhs, int[] ds, int[] widx,
                           int[] tk, int[] nk) {

    int len = ds[d + 1] - ds[d];

    if (len == 0)
      return 0;

    double ll = 0;
    int si = ds[d];
    int ei = ds[d + 1];
    short tt, ttt;
    int idx;

    Arrays.fill(tk, 0);
    for (int di = si; di < ei; di++) {
      tk[topics[widx[di]]]++;
    }

    // traverse d first time, accept word proposal
    for (int di = si; di < ei; di++) {
      idx = widx[di];

      tt = topics[idx];
      ttt = mhs[idx];

      if (tt != ttt) {
        nk[tt]--;
        tk[tt]--;

        float p = ((tk[ttt] + alpha) * (nk[tt] + vbeta)) / ((tk[tt] + alpha) * (nk[ttt] + vbeta));
        if (rand.nextFloat() < p) {
          tt = ttt;
        }

        nk[tt]++;
        tk[tt]++;
        topics[idx] = tt;
      }
    }

    // traverse d second time, generate doc proposal
    float p = (alpha * K) / (alpha * K + len);
    for (int di = si; di < ei; di++) {

      if (rand.nextFloat() < p) {
        mhs[widx[di]] = (short) rand.nextInt(K);
      } else {
        mhs[widx[di]] = topics[widx[ds[d] + rand.nextInt(len)]];
      }

    }

    if (test) {

      for (int k = 0; k < K; k++)
        if (tk[k] > 0)
          ll += Gamma.logGamma(alpha + tk[k]) - lgamma_alpha;

      double value = Gamma.logGamma(alpha * K + len) - Gamma.logGamma(alpha * K);
      ll -= value;
    }
//    System.out.format("N=%d value=%f\n", len, value);
    return ll;
  }

  public double visitByCol(int w, Random rand, short[] topics, short[] mhs, int[] ws, int[] tk, int[] nk) {
    int len = ws[w + 1] - ws[w];

    if (len == 0)
      return 0;

    double ll = 0;
    int si = ws[w];
    int ei = ws[w + 1];
    short tt, ttt;
    int idx;

    Arrays.fill(tk, 0);
    for (int wi = si; wi < ei; wi++)
      tk[topics[wi]]++;

    for (int wi = si; wi < ei; wi++) {
      tt = topics[wi];
      ttt = mhs[wi];

      if (tt != ttt) {

        tk[tt]--;
        nk[tt]--;

        float b = tk[tt] + beta;
        float d = nk[tt] + vbeta;
        float a = tk[ttt] + beta;
        float c = nk[ttt] + vbeta;

        if (rand.nextFloat() < (a * d) / (b * c)) {
          tt = ttt;
        }

        topics[wi] = tt;
        tk[tt]++;
        nk[tt]++;
      }
    }

    // traverse two time, compute word proposal
    float p = (K * beta) / (K * beta + len);

    for (int wi = si; wi < ei; wi++) {
      float u = rand.nextFloat();
      if (u < p) {
        mhs[wi] = (short) rand.nextInt(K);
      } else {
        idx = rand.nextInt(len);
        mhs[wi] = topics[ws[w] + idx];
      }
    }

    if (test) {
      for (int k = 0; k < K; k++)
        if (tk[k] > 0)
          ll += Gamma.logGamma(beta + tk[k]) - lgamma_beta;
    }

    return ll;
  }

  public void reduceNk(Task[] tasks) {
    for (int i = 0; i < taskNum; i++) {
      int[] tnk = tasks[i].nk;
      for (int k = 0; k < K; k++) {
        tnk[k] -= nk[k];
      }
    }

    for (int i = 0; i < taskNum; i++) {
      int[] tnk = tasks[i].nk;
      for (int k = 0; k < K; k++) {
        nk[k] += tnk[k];
      }
    }
  }

  public double waitForFutures() {
    double sum = 0;
    for (int i = 0; i < taskNum; i++) {
      try {
        sum += futures[i].get();
      } catch (InterruptedException e) {
        e.printStackTrace();
      } catch (ExecutionException e) {
        e.printStackTrace();
      }
    }
    return sum;
  }

  public void init() {
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w++)
      queue.add(w);

    for (int i = 0; i < taskNum; i++) {
      initTasks[i].set(queue, nk);
      futures[i] = executor.submit(initTasks[i]);
    }

    waitForFutures();
    reduceNk(initTasks);
  }

  public double visitByRow() {

    long start = System.currentTimeMillis();
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int d = 0; d < D; d++)
      queue.add(d);

    for (int i = 0; i < taskNum; i++) {
      rowTasks[i].set(queue, nk);
      futures[i] = executor.submit(rowTasks[i]);
    }

    long tt = System.currentTimeMillis() - start;

    double ll = waitForFutures();
    reduceNk(rowTasks);
    return ll;
  }

  public double visitByCol() {
    long start = System.currentTimeMillis();
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w++)
      queue.add(w);

    for (int i = 0; i < taskNum; i++) {
      colTasks[i].set(queue, nk);
      futures[i] = executor.submit(colTasks[i]);
    }

    long tt = System.currentTimeMillis() - start;

    double ll = waitForFutures();
    reduceNk(colTasks);

    if (test) {
      ll += K * Gamma.logGamma(vbeta);
      for (int k = 0; k < K; k++) {
        ll -= Gamma.logGamma(nk[k] + vbeta);
      }
    }

    return ll;
  }

  public void train(int it) {

    if (it % 10 == 0)
      test = true;
    else
      test = false;

    long start;
    long row_tt, col_tt;
    start = System.currentTimeMillis();
    double ll = visitByRow();
    row_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    ll += visitByCol();
    col_tt = System.currentTimeMillis() - start;
    System.out.format("it=%d ll=%f row_tt=%d col_tt=%d iter_tt=%d\n",
            it, ll, row_tt, col_tt, row_tt + col_tt);
  }

  public void trainJni(int it) {
//    double ll = lda.jni.Utils.warpOneIter(it, D, V, K, mat.N,
//      mat.ws, mat.ds, mat.widx, nk, mat.topics0, mat.mh, MH_STEPS,
//      alpha, beta, vbeta, maxlen);

//    System.out.format("it=%d ll=%f\n", it, ll);
  }


  abstract class Task implements Callable<Double> {

    public int[] nk;
    public int[] tk;
    public ConcurrentLinkedQueue<Integer> queue;

    public Task() {
      tk = new int[K];
      nk = new int[K];
    }

    public void set(ConcurrentLinkedQueue<Integer> queue, int[] nk) {
      this.queue = queue;
      System.arraycopy(nk, 0, this.nk, 0, K);
    }

    public int[] getNk() {
      return nk;
    }
  }

  class RowTask extends Task {
    @Override
    public Double call() throws Exception {
      Integer d;
      Random rand = new Random(System.currentTimeMillis());
      int[] ds = mat.ds;
      int[] widx = mat.widx;
      short[] topics = mat.topics0;
      short[] mhs = mat.mh;
      double ll = 0;
      while ((d = queue.poll()) != null) {
        ll += visitByRow(d, rand, topics, mhs, ds, widx, tk, nk);
      }
      return ll;
    }
  }

  class ColTask extends Task {
    @Override
    public Double call() throws Exception {
      Integer w;
      Random rand = new Random(System.currentTimeMillis());
      int[] ws = mat.ws;
      short[] topics = mat.topics0;
      short[] mhs = mat.mh;
      double ll = 0;
      while ((w = queue.poll()) != null) {
        ll += visitByCol(w, rand, topics, mhs, ws, tk, nk);
      }
      return ll;
    }
  }

  class InitTask extends Task {

    @Override
    public Double call() throws Exception {
      Integer w;
      int[] ws = mat.ws;
      short[] topics = mat.topics0;
      short[] mhs = mat.mh;
      Random rand = new Random(System.currentTimeMillis());
      while ((w = queue.poll()) != null) {
        init(w, rand, ws, mhs, topics, nk);
      }
      return 0.0;
    }
  }

  public static void nips(String[] argv) throws IOException {
//    String path = "docword.nips.txt";
    String path = "nips.train";
    int V = 12420;
    int K = 1024;
    DVMat mat = new DVMat(path, V);
    float alpha = 50.0F / K;
    float beta = 0.01F;

    WarpLDA lda = new WarpLDA(mat.D, mat.V, K, alpha, beta, mat, 1);
    long start = System.currentTimeMillis();
    lda.init();
    long end = System.currentTimeMillis();

    System.out.format("init_tt=%d\n", end - start);

    for (int i = 0; i < 200; i++) {
//      lda.train(i);
      lda.trainJni(i);
    }
    lda.executor.shutdown();
  }

  public static void cmds(String[] argv) throws IOException {
    String path = argv[0];
    int V = Integer.parseInt(argv[1]);
    int K = 1024;
    DVMat mat = new DVMat(path, V);
    System.out.println("finish reading " + mat.D + " documents");
    float alpha = 50.0F / K;
    float beta = 0.01F;
    int taskNum = Integer.parseInt(argv[2]);

    WarpLDA lda = new WarpLDA(mat.D, mat.V, K, alpha, beta, mat, taskNum);
    long start = System.currentTimeMillis();
    lda.init();
    long end = System.currentTimeMillis();

    System.out.format("init_tt=%d\n", end - start);

    int I = Integer.parseInt(argv[3]);
    for (int i = 0; i < I; i++) {
      lda.train(i);
    }
    lda.executor.shutdown();
  }

  public static void main(String[] argv) throws IOException {
    System.loadLibrary("warp_jni");
    nips(argv);
//    cmds(argv);
  }


}
