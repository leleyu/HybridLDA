package lda.parallel;

import lda.utils.BinarySearch;
import lda.utils.Document;
import lda.utils.Utils;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by leleyu on 2017/2/9.
 */
public class AliasLDADoc {

  private Alias[] alias;
  private final int MH_STEPS = 1;

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public int[][] nwk;
  public int[] nk;

  public int[] ds;
  public int[] wids;
  public short[] topics;

  public float[] p;
  public float[] qws;

  public double lgammaBeta;
  public double lgammaAlpha;
  public double lgammaAlphaSum;

  public ThreadPoolExecutor executor;
  private boolean test = false;
  public int taskNum;

  public Task[] initTasks;
  public Task[] sampleTasks;
  public Task[] buildTasks;
  public Task[] testTasks;
  public Future<Double>[] futures;

  public AliasLDADoc(int D, int V, int K, float alpha, float beta, List<Document> docs,
                     int taskNum) {

    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta = beta;
    this.vbeta = V * beta;

    buildMat(docs);

    lgammaBeta = Gamma.logGamma(beta);
    lgammaAlpha = Gamma.logGamma(alpha);
    lgammaAlphaSum = Gamma.logGamma(alpha * K);

    this.taskNum = taskNum;
    initTasks = new InitTask[taskNum];
    sampleTasks = new SampleTask[taskNum];
    buildTasks = new BuildTask[taskNum];
    testTasks  = new TestTask[taskNum];
    futures = new Future[taskNum];

    for (int i = 0; i < taskNum; i ++) {
      initTasks[i] = new InitTask();
      sampleTasks[i] = new SampleTask();
      buildTasks[i] = new BuildTask();
      testTasks[i] = new TestTask();
    }

    executor = new ThreadPoolExecutor(16, 32, 60,
            TimeUnit.HOURS,
            new LinkedBlockingQueue<Runnable>());
  }

  public void buildMat(List<Document> docs) {

    Document doc;
    N = 0;

    ds = new int[D + 1];
    ds[0] = 0;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      N += doc.wids.length;
      ds[d + 1] = N;
    }

    wids = new int[N];
    topics = new short[N];
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int i = 0; i < doc.wids.length; i ++) {
        wids[ds[d] + i] = doc.wids[i];
      }
    }
  }

  public void reduceNk(Task[] tasks) {
    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = tasks[i].nk;
      for (int k = 0; k < K; k ++) {
        localNk[k] -= nk[k];
      }
    }

    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = tasks[i].nk;
      for (int k = 0; k < K; k ++)
        nk[k] += localNk[k];
    }
  }

  public double waitForFutures() {
    double sum = 0.0;
    try {
      for (int i = 0; i < taskNum; i ++)
        sum += futures[i].get();
    } catch (InterruptedException e) {
      e.printStackTrace();
    } catch (ExecutionException e) {
      e.printStackTrace();
      e.getCause().printStackTrace();
    }
    return sum;
  }

  public void init(int d, Random rand, int[] nk) {
    for (int j = ds[d]; j < ds[d + 1]; j ++) {
      short tt = (short) rand.nextInt(K);
      topics[j] = tt;
      nk[tt] ++;
      int wid = wids[j];
      synchronized (nwk[wid]) {
        nwk[wid][tt]++;
      }
    }
  }

  public void build(int wid, Alias alias, int[] wk) {
    float[] qw = alias.probability;
    float Qw = 0.0F;
    for (int k = 0; k < K; k ++) {
      qw[k] = (wk[k] + beta) / (nk[k] + vbeta);
      Qw += qw[k];
    }

    qws[wid] = Qw;

    for (int k = 0; k < K; k ++)
      qw[k] /= Qw;

    alias.build(qw);
  }

  public double trainOneDoc(int d, int[] dk, short[] didx, short[] poss, int[] nk, float[] p) {
    Random rand = new Random(System.currentTimeMillis());

    int wid;
    short tt, ttt;
    int kk, vv;
    Alias alias1;

    short size = 0;
    // Calculate dk
    Arrays.fill(dk, 0);
    for (int j = ds[d]; j < ds[d + 1]; j ++) {
      tt = topics[j];
      dk[tt] ++;
      if (dk[tt] == 1) {
        poss[tt] = size;
        didx[size++] = tt;
      }
    }

    for (int j = ds[d]; j < ds[d + 1]; j ++) {

      wid = wids[j];
      tt  = topics[j];

      alias1 = alias[wid];


      nk[tt] --;
      dk[tt] --;

      if (dk[tt] == 0) {
        size --;
        didx[poss[tt]] = didx[size];
        poss[didx[size]] = poss[tt];
      }

      synchronized (nwk[wid]) {
        nwk[wid][tt]--;

        // Compute pdw
        float psum = 0.0F;

        for (int i = 0; i < size; i++) {
          kk = didx[i];
          vv = dk[kk];
          psum += vv * (nwk[wid][kk] + beta) / (nk[kk] + vbeta);
          p[i] = psum;
        }

        // probability to select pdw
        float select_pr = psum / (psum + alpha * qws[wid]);

        // MHV to draw a new topic
        for (int r = 0; r < MH_STEPS; r++) {
          // flip a corn to choose pdw or qw
          // rand from uniform01

          if (rand.nextFloat() < select_pr) {
            // sample from pdm
            float u = rand.nextFloat() * psum;
            int idx = BinarySearch.binarySearch(p, u, 0, size - 1);
            ttt = didx[idx];
//          ttt = dk.key[dk.idx[idx]];
          } else {
            // sample from qw
            ttt = (short) alias1.next();
          }

          if (tt != ttt) {
            // compute acceptance probability

            int s_n_dk = dk[tt];
            int t_n_dk = dk[ttt];

            double s_qw = alias1.qw[tt];
            double t_qw = alias1.qw[ttt];

            double temp_s = (nwk[wid][tt] + beta) / (nk[tt] + vbeta);
            double temp_t = (nwk[wid][ttt] + beta) / (nk[ttt] + vbeta);
            double acceptance = (t_n_dk + alpha) / (s_n_dk + alpha)
                    * temp_t / temp_s
                    * (s_n_dk * temp_s + alpha * s_qw * qws[wid])
                    / (t_n_dk * temp_t + alpha * t_qw * qws[wid]);

            // compare against uniform[0,1]
            if (rand.nextFloat() < acceptance) {
              tt = ttt;
            }
          }
        }

        nwk[wid][tt]++;
      }

      topics[j] = tt;


      dk[tt] ++;
      if (dk[tt] == 1) {
        didx[size] = tt;
        poss[tt] = size;
        size ++;
      }
      nk[tt] ++;
    }

    double ll = 0;
    if (test) {
      int len = ds[d + 1] - ds[d];
      ll += lgammaAlphaSum - Gamma.logGamma(alpha * K + len);
      for (int i = 0; i < K; i++) {
        if (dk[i] != 0)
          ll += Gamma.logGamma(alpha + dk[i]) - lgammaAlpha;
      }
    }

    return ll;
  }

  public double computeWordLLH(int wid) {
    int[] wk = nwk[wid];
    double ll = 0;
    for (int k = 0; k < K; k ++)
      if (wk[k] > 0)
        ll += Gamma.logGamma(wk[k] + beta) - lgammaBeta;
    return ll;
  }

  public double computeWordLLHSummary() {
    double ll = 0.0;
    ll += K * Gamma.logGamma(beta * V);
    for (int k = 0; k < K; k ++) {
      ll -= Gamma.logGamma(nk[k] + beta * V);
    }
    return ll;
  }

  abstract class Task implements Callable<Double> {

    public int[] nk;
    public int[] tk;
    public AtomicInteger id;

    public Task() {
      tk = new int[K];
      nk = new int[K];
    }

    public void set(AtomicInteger id, int[] nk) {
      this.id = id;
      System.arraycopy(nk, 0, this.nk, 0, K);
    }

    public int[] getNk() {
      return nk;
    }
  }

  class InitTask extends Task {
    @Override
    public Double call() throws Exception {
      Random rand = new Random(System.currentTimeMillis());
      while (true) {
        int did = id.getAndIncrement();
        if (did < D) {
          init(did, rand, nk);
        } else {
          break;
        }
      }
      return 0.0;
    }
  }

  class BuildTask extends Task {
    @Override
    public Double call() throws Exception {
      while (true) {
        int wid = id.getAndIncrement();
        if (wid < V) {
          build(wid, alias[wid], nwk[wid]);
        } else {
          break;
        }
      }
      return 0.0;
    }
  }

  class SampleTask extends Task {

    short[] didx;
    short[] poss;
    float[] p;
    public SampleTask() {
      super();
      this.didx = new short[K];
      this.poss = new short[K];
      this.p    = new float[K];
    }

    @Override
    public Double call() throws Exception {
      double ll = 0;
      while (true) {
        int did = id.getAndIncrement();
        if (did < D) {
          ll += trainOneDoc(did, tk, didx, poss, nk, p);
        } else {
          break;
        }
      }
      return ll;
    }
  }

  class TestTask extends Task {
    @Override
    public Double call() throws Exception {
      double ll = 0;
      while (true) {
        int wid = id.getAndIncrement();
        if (wid < V) {
          ll += computeWordLLH(wid);
        } else {
          break;
        }
      }
      return ll;
    }
  }

  public void init() {
    alias = new Alias[V];

    nwk = new int[V][K];
    nk  = new int[K];

    p = new float[K];
    qws = new float[V];

    for (int w = 0; w < V; w ++) {
      nwk[w] = new int[K];
      alias[w] = new Alias(K);
    }

    AtomicInteger did = new AtomicInteger(0);
    for (int i = 0; i < taskNum; i ++) {
      initTasks[i].set(did, nk);
      futures[i] = executor.submit(initTasks[i]);
    }

    waitForFutures();
    reduceNk(initTasks);
  }

  public void build() {
    AtomicInteger wid = new AtomicInteger(0);
    for (int i = 0; i < taskNum; i ++) {
      buildTasks[i].set(wid, nk);
      futures[i] = executor.submit(buildTasks[i]);
    }

    waitForFutures();
  }

  public double train(int it) {
    AtomicInteger did = new AtomicInteger(0);
    for (int i = 0; i < taskNum; i ++) {
      sampleTasks[i].set(did, nk);
      futures[i] = executor.submit(sampleTasks[i]);
    }

    double ll = waitForFutures();
    reduceNk(sampleTasks);
    return ll;
  }

  public double test(int it) {
    AtomicInteger wid = new AtomicInteger(0);
    for (int i = 0; i < taskNum; i ++) {
      testTasks[i].set(wid, nk);
      futures[i] = executor.submit(testTasks[i]);
    }



    return waitForFutures() + computeWordLLHSummary();
  }


  public void iteration(int it) {
    if (it % 5 == 0)
      test = true;
    else
      test = false;

    long start;
    long train_tt, eval_tt, build_tt;

    start = System.currentTimeMillis();
    build();
    build_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    double ll = train(it);
    train_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    if (test) {
      ll += test(it);
    }
    eval_tt = System.currentTimeMillis() - start;

    System.out.format("it=%d ll=%f train_tt=%d eval_tt=%d build_tt=%d\n",
            it, ll, train_tt, eval_tt, build_tt);
  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int M = docs.size();
    int V = 12420;
    int K = 1024;
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    AliasLDADoc lda = new AliasLDADoc(M, V, K, alpha, beta, docs, 1);
    lda.init();

    for (int i = 0; i < 100; i ++) {
      lda.iteration(i);
    }
  }

  public static AliasLDADoc read(String[] argv) throws IOException {
    String path = argv[0];
    List<Document> docs = Utils.read(path);
    int M = docs.size();
    int V = Integer.parseInt(argv[1]);
    int K = Integer.parseInt(argv[2]);
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    int taskNum = Integer.parseInt(argv[3]);
    int iter = Integer.parseInt(argv[4]);

    System.out.format("D=%d V=%d K=%d alpha=%f beta=%f iter=%d\n",
            M, V, K, alpha, beta, iter);

    AliasLDADoc lda = new AliasLDADoc(M, V, K, alpha, beta, docs, taskNum);
    return lda;
  }

  public static void cmds(String[] argv) throws IOException {
    AliasLDADoc lda = read(argv);
    lda.init();
    int iter = Integer.parseInt(argv[4]);
    for (int i = 0; i < iter; i ++)
      lda.iteration(i);

    lda.executor.shutdown();
  }

  public static void main(String[] argv) throws IOException {
//    nips(argv);
    cmds(argv);
  }
}
