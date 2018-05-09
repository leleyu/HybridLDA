package lda.parallel;

import lda.sparse.*;
import lda.utils.*;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Created by leleyu on 2016/9/30.
 */
public class AliasLDA {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public CsrMat mat;
  public short[] topics;
  public int[] nk;

  public int taskNum;

  TraverseHashMap[] ndk;
  public int[] docLens;
  public int MH_STEPS = 1;

  SampleTask[] sampleTasks;
  TestTask[] testTasks;
  InitTask[] initTasks;
  Future<Double>[] futures;

  public double lgammaBeta;
  public double lgammaAlpha;
  public double lgammaAlphaSum;

  public ThreadPoolExecutor executor;

  private boolean test = false;

  public AliasLDA(int D, int V, int K, float alpha, float beta,
                  List<Document> docs, int taskNum) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta  = beta;
    this.vbeta = V * beta;

    docLens = new int[D];

    buildMat(docs);

    topics = new short[N];
    nk = new int[K];
    ndk = new TraverseHashMap[D];

    initDk();

    this.taskNum = taskNum;
    sampleTasks = new SampleTask[taskNum];
    testTasks   = new TestTask  [taskNum];
    initTasks   = new InitTask  [taskNum];
    futures     = new Future[taskNum];
    for (int i = 0; i < taskNum; i ++) {
      sampleTasks[i] = new SampleTask();
      testTasks  [i] = new TestTask();
      initTasks  [i] = new InitTask();
    }

    lgammaBeta = Gamma.logGamma(beta);
    lgammaAlpha = Gamma.logGamma(alpha);
    lgammaAlphaSum = Gamma.logGamma(alpha * K);

    executor = new ThreadPoolExecutor(16, 32, 60,
            TimeUnit.HOURS,
            new LinkedBlockingQueue<Runnable>());
  }

  public void initDk() {
    int a = 0, b = 0, c = 0;

    for (int d = 0; d < D; d ++) {

      if (docLens[d] < Byte.MAX_VALUE) {
        ndk[d] = new S2BTraverseMap(docLens[d]);
        a ++;
        continue;
      }
      else if (docLens[d] < (K / 2)) {
        ndk[d] = new S2STraverseMap(Math.min(K, docLens[d]));
        b ++;
        continue;
      } else {
        ndk[d] = new S2STraverseArray(K);
        c ++;
        continue;
      }
    }

    System.out.format("a=%d b=%d c=%d\n", a, b, c);
  }

  public void init(int wid, Random rand, int[] nk) {
    int si = mat.rs[wid];
    int ei = mat.rs[wid + 1];
    for (int wi = si; wi < ei; wi ++) {
      short kk = (short) rand.nextInt(K);
      topics[wi] = kk;
      nk[kk] ++;
      int d = mat.cols[wi];
      synchronized (ndk[d]) {
        ndk[d].inc(kk);
      }
    }
  }

  public void buildMat(List<Document> docs) {
    Document doc;
    int[] wcnt = new int[V];
    int[] ws = new int[V + 1];

    N = 0;
    // count word
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int w = 0; w < doc.wids.length; w ++)
        wcnt[doc.wids[w]] ++;
      docLens[d] = doc.length;
      N += doc.length;
    }

    int[] cols = new int[N];

    // build word start idx
    ws[0] = 0;
    for (int i = 0; i < V; i ++) {
      ws[i + 1] = ws[i] + wcnt[i];
    }

    for (int d = D - 1; d >= 0; d --) {
      doc = docs.get(d);
      int wid;
      for (int w = 0; w < doc.wids.length; w ++) {
        wid = doc.wids[w];
        int pos = ws[wid] + (--wcnt[wid]);
        cols[pos] = d;
      }
    }

    mat = new CsrMat(V, D, ws, cols);
  }

  public float buildAliasTable(int[] wk, Alias alias, int[] small, int[] large) {
    float Qw = 0.0F;
    float[] qw = alias.probability;

    for (int k = 0; k < K; k ++) {
      qw[k] = (wk[k] + beta) / (nk[k] + vbeta);
      Qw += qw[k];
    }

    for (int k = 0; k < K; k ++)
      qw[k] /= Qw;

    alias.build(qw, small, large);
    return Qw;
  }

  public float buildDocDist(int did, float[] p, short[] tidx, int[] wk) {
    TraverseHashMap map = ndk[did];
    float psum = 0.0F;
    short topic;
    short count;
    int idx = 0;

    if (map instanceof S2STraverseArray) {
      S2STraverseArray buf = (S2STraverseArray) map;
      for (int i = 0; i < buf.size; i ++) {
        topic = buf.idx[i];
        count = buf.value[topic];
        psum += count * (wk[topic] + beta) / (nk[topic] + vbeta);
        p[idx] = psum;
        tidx[idx ++] = topic;
      }

      return psum;
    }

    if (map instanceof S2STraverseMap) {
      S2STraverseMap buf = (S2STraverseMap) map;

      for (int i = 0; i < buf.size; i ++) {
        topic = buf.key[buf.idx[i]];
        count = buf.value[buf.idx[i]];
        psum += count * (wk[topic] + beta) / (nk[topic] + vbeta);
        p[idx] = psum;
        tidx[idx ++] = topic;
      }

      return psum;
    }

    if (map instanceof S2BTraverseMap) {
      S2BTraverseMap buf = (S2BTraverseMap) map;
      for (int i = 0; i < buf.size; i ++) {
        topic = buf.key[buf.idx[i]];
        count = buf.value[buf.idx[i]];
        psum += count * (wk[topic] + beta) / (nk[topic] + vbeta);
        p[idx] = psum;
        tidx[idx ++] = topic;
      }
      return psum;
    }

    for (int i = 0; i < map.size; i ++) {
      topic = map.getKey(i);
      count = map.getVal(i);
      psum += count * (wk[topic] + beta) / (nk[topic] + vbeta);
      p[idx] = psum;
      tidx[idx ++] = topic;
    }
    return psum;
  }

  public void sample(int wid, Alias alias, float[] p, int[] nk, int[] wk, int[] small, int[] large, short[] tidx) {
    Random rand = new Random(System.currentTimeMillis());
    float Qw = buildAliasTable(wk, alias, small, large);

    short tt, ttt;
    short kk, vv;
    float psum, u;

    int si, ei, d;
    si = mat.rs[wid];
    ei = mat.rs[wid + 1];

    TraverseHashMap dk;
    int[] tokens = mat.cols;

    for (int wi = si; wi < ei; wi++) {
      d = tokens[wi];
      dk = ndk[d];
      tt = topics[wi];

      wk[tt]--;
      nk[tt]--;

      synchronized (dk) {
        dk.dec(tt);

        psum = buildDocDist(d, p, tidx, wk);
        float select_pr = psum / (psum + alpha * Qw);

        for (int r = 0; r < MH_STEPS; r++) {

          if (rand.nextFloat() < select_pr) {
            u = rand.nextFloat() * psum;
            ttt = tidx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)];
          } else {
            ttt = (short) alias.next();
          }

          if (tt != ttt) {
            int s_n_dk = dk.get(tt);
            int t_n_dk = dk.get(ttt);

            float s_qw = alias.qw[tt];
            float t_qw = alias.qw[ttt];

            float temp_s = (wk[tt] + beta) / (nk[tt] + vbeta);
            float temp_t = (wk[ttt] + beta) / (nk[ttt] + vbeta);
            float acceptance = (t_n_dk + alpha) / (s_n_dk + alpha)
                    * temp_t / temp_s
                    * (s_n_dk * temp_s + alpha * s_qw * Qw)
                    / (t_n_dk * temp_t + alpha * t_qw * Qw);

            if (rand.nextFloat() < acceptance)
              tt = ttt;
          }
        }

        dk.inc(tt);
      }
      wk[tt]++;
      nk[tt]++;
      topics[wi] = tt;
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

  public double train(int it) {
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w ++)
      queue.add(w);

    for (int t = 0; t < taskNum; t ++) {
      sampleTasks[t].set(queue, nk);
      futures[t] = executor.submit(sampleTasks[t]);
    }

    double ll = waitForFutures();
    reduceNk(sampleTasks);
    return ll;
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
    }
    return sum;
  }

  public double computeDocLLH() {

    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int d = 0; d < D; d ++)
      queue.add(d);

    for (int i = 0; i < taskNum; i ++) {
      testTasks[i].set(queue, nk);
      futures[i] = executor.submit(testTasks[i]);
    }

    double ll = waitForFutures();

    return ll;
  }

  public double computeDocLLH(int d, TraverseHashMap dk) {
    double ll = 0;
    for (int j = 0; j < dk.size; j++) {
      short count = dk.getVal(j);
      ll += Gamma.logGamma(alpha + count) - lgammaAlpha;
    }
    ll -= Gamma.logGamma(alpha * K + docLens[d]) - lgammaAlphaSum;
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

  public void iteration(int it) {
    if (it % 5 == 0)
      test = true;
    else
      test = false;

    long start;
    long train_tt, eval_tt;
    start = System.currentTimeMillis();
    double ll = train(it);
    train_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    if (test) {
      ll += computeDocLLH();
      ll += computeWordLLHSummary();
    }
    eval_tt = System.currentTimeMillis() - start;
    System.out.format("it=%d ll=%f train_tt=%d eval_tt=%d\n",
            it, ll, train_tt, eval_tt);
  }

  public void init() {
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w ++)
      queue.add(w);

    for (int i = 0; i < taskNum; i ++) {
      initTasks[i].set(queue, nk);
      futures[i] = executor.submit(initTasks[i]);
    }

    waitForFutures();
    reduceNk(initTasks);
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

  class InitTask extends Task {
    @Override
    public Double call() throws Exception {
      Integer w;
      Random rand = new Random(System.currentTimeMillis());
      while ((w = queue.poll()) != null) {
        init(w, rand, nk);
      }

      return 0.0;
    }
  }

  class SampleTask extends Task {
    public float[] p;
    public short[] tidx;
    public int[] wk;
    public Alias alias;
    public int[] small;
    public int[] large;

    public SampleTask() {
      p = new float[K];
      tidx = new short[K];
      wk = new int[K];
      alias = new Alias(K);
      small = new int[K];
      large = new int[K];
    }

    @Override
    public Double call() throws Exception {

      Integer w;
      double ll = 0;
      while ((w = queue.poll()) != null) {

        Arrays.fill(wk, 0);
        for (int wi = mat.rs[w]; wi < mat.rs[w + 1]; wi ++)
          wk[topics[wi]] ++;

        sample(w, alias, p, nk, wk, small, large, tidx);

        if (test) {
          for (int k = 0; k < K; k++)
            if (wk[k] > 0) {
              ll += Gamma.logGamma(wk[k] + beta) - lgammaBeta;
            }
        }
      }

      return ll;
    }
  }

  class TestTask extends Task {
    @Override
    public Double call() throws Exception {
      Integer d;
      double ll = 0;
      while ((d = queue.poll()) != null) {
        ll += computeDocLLH( d, ndk[d]);
      }
      return ll;
    }
  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int M = docs.size();
    int V = 12420;
    int K = 1024;
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    AliasLDA lda = new AliasLDA(M, V, K, alpha, beta, docs, 1);
    lda.init();

    for (int i = 0; i < 100; i ++) {
      lda.iteration(i);
    }
  }

  public static AliasLDA read(String[] argv) throws IOException {
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

    AliasLDA lda = new AliasLDA(M, V, K, alpha, beta, docs, taskNum);
    return lda;
  }

  public static void cmds(String[] argv) throws IOException {
    AliasLDA lda = read(argv);
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
