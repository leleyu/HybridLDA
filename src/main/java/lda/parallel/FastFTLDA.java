package lda.parallel;

import lda.utils.*;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;


/**
 * Created by leleyu on 2016/3/9.
 */
public class FastFTLDA {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public CsrMat mat;
  public short[] topics;
  public int[] nk;

  public int taskNum;

  TraverseMap[] ndk;
  public int[] docLens;

  SampleTask[] sampleTasks;
  TestTask[] testTasks;
  InitTask[] initTasks;
  Future<Double>[] futures;

  public double lgammaBeta;
  public double lgammaAlpha;
  public double lgammaAlphaSum;

  public ThreadPoolExecutor executor;



  public FastFTLDA(int D, int V, int K, float alpha, float beta,
                  List<Document> docs, int taskNum) {

    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta  = beta;
    this.vbeta = V * beta;

    docLens = new int[D];

    buildMat(docs);

    System.out.format("# tokens = " + N + "\n");

    topics = new short[N];
    nk = new int[K];
    ndk = new TraverseMap[D];

    for (int d = 0; d < D; d ++)
      ndk[d] = new TraverseMap(Math.min(K, docLens[d]));

    this.taskNum = taskNum;
    sampleTasks = new SampleTask[taskNum];
    testTasks   = new TestTask  [taskNum];
    initTasks   = new InitTask  [taskNum];
    futures     = new Future[taskNum];
    for (int i = 0; i < taskNum; i ++) {
      sampleTasks[i] = new SampleTask();
      testTasks  [i] = new TestTask();
    }

    lgammaBeta = Gamma.logGamma(beta);
    lgammaAlpha = Gamma.logGamma(alpha);
    lgammaAlphaSum = Gamma.logGamma(alpha * K);

    executor = new ThreadPoolExecutor(16, 32, 60,
            TimeUnit.HOURS,
            new LinkedBlockingQueue<Runnable>());
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

  public void buildFTree(int[] wk, FTree tree, float[] p, int[] nk) {
    for (int topic = 0; topic < K; topic ++) {
      p[topic] = (wk[topic] + beta) / (nk[topic] + vbeta);
    }

    tree.build(p);
  }

  public float buildDocDist(int did, float[] p, short[] tidx, FTree tree) {
    float psum = 0.0F;
    TraverseMap map = ndk[did];
    short topic;
    short count;
    int idx = 0;
    for (int i = 0; i < map.size; i ++) {
      topic = map.key[map.idx[i]];
      count = map.value[map.idx[i]];
      psum += count * tree.get(topic);
      p[idx] = psum;
      tidx[idx ++] = topic;
    }
    return psum;
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

  public void init() {
    Random rand = new Random(System.currentTimeMillis());
    for (int w = 0; w < V; w ++) {
      init(w, rand, null);
    }
  }

  public void initParallel() {
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w ++)
      queue.add(w);

    for (int i = 0; i < taskNum; i ++) {
      initTasks[i] = new InitTask(queue);
      futures[i] = executor.submit(initTasks[i]);
    }

    waitForFutures();
    reduceNkInit();
  }

  public void reduceNkInit() {
    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = initTasks[i].nk;
      for (int k = 0; k < K; k ++)
        nk[k] += localNk[k];
    }
  }

  public double norm(int[] wk) {
    double sum = 0.0, x;
    for (int k = 0; k < K; k ++) {
      x = wk[k] + beta;
      sum += x * x;
    }
    return sum;
  }

  public void sample(int wid, FTree tree, float[] p, short[] tidx, int[] nk, int[] wk, Metric metric) {
    Random random = new Random(System.currentTimeMillis());

    float value;
    int si, ei;
    si = mat.rs[wid];
    ei = mat.rs[wid + 1];
    int[] tokens = mat.cols;
    int d, size, idx;
    short kk;
    float psum, u;
    TraverseMap dk;

    double b = norm(wk);
    double zt;

    for (int wi = si; wi < ei; wi ++) {
      d = tokens[wi];
      dk = ndk[d];
      kk = topics[wi];

      wk[kk] --;
      nk[kk] --;
      value = (wk[kk] + beta) / (nk[kk] + vbeta);
      tree.update(kk, value);

      b += - 1.0 - 2.0 * (wk[kk] + beta);
      u = random.nextFloat();

      synchronized (dk) {
        dk.dec(kk);

        psum = alpha * tree.first();
        zt = psum + Math.sqrt(dk.norm * b) / (nk[0] + vbeta);

        if (psum >= u * zt) {
          kk = (short) tree.sample(random.nextFloat() * tree.first());
          metric.num1 ++;
        } else {
          size = dk.size;
          psum = buildDocDist(d, p, tidx, tree);
          u = random.nextFloat() * (psum + alpha * tree.first());

          if (u < psum) {
            u = random.nextFloat() * psum;
            idx = BinarySearch.binarySearch(p, u, 0, size - 1);
            kk = tidx[idx];
          } else {
            kk = (short) tree.sample(random.nextFloat() * tree.first());
            metric.num2 ++;
          }
        }


        dk.inc(kk);
      }

      wk[kk] ++;
      nk[kk] ++;
      value = (wk[kk] + beta) / (nk[kk] + vbeta);
      tree.update(kk, value);
      topics[wi] = kk;
      b += - 1.0 + 2.0 * (wk[kk] + beta);
    }
  }

  public double trainParallel(int it) {
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w ++)
      queue.add(w);

    for (int t = 0; t < taskNum; t ++) {
      sampleTasks[t].setTaskQueue(queue, nk);
      futures[t] = executor.submit(sampleTasks[t]);
    }

    double ll = waitForFutures();
    reduceNk();

    long num1 = 0;
    long num2 = 0;
    for (int i = 0; i < taskNum; i ++) {
      Metric metric = sampleTasks[i].metric;
      num1 += metric.num1;
      num2 += metric.num2;
    }

    System.out.format("num1=%d num2=%d\n", num1, num2);

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

  public void reduceNk() {
    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = sampleTasks[i].nk;
      for (int k = 0; k < K; k ++) {
        localNk[k] -= nk[k];
      }
    }

    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = sampleTasks[i].nk;
      for (int k = 0; k < K; k ++)
        nk[k] += localNk[k];
    }
  }

  public double loglikelihood() {
    double ll = computeDocLLH();
    ll += computeWordLLHSummary();
    return ll;
  }

  public void iteration(int it) {
    long start;
    long train_tt, eval_tt;
    start = System.currentTimeMillis();
    double ll = trainParallel(it);
    train_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    ll += loglikelihood();

    eval_tt = System.currentTimeMillis() - start;
    System.out.format("it=%d ll=%f train_tt=%d eval_tt=%d\n",
            it, ll, train_tt, eval_tt);
  }

  public double computeDocLLH() {

    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int d = 0; d < D; d ++)
      queue.add(d);

    for (int i = 0; i < taskNum; i ++) {
      testTasks[i].setTaskQueue(queue);
      futures[i] = executor.submit(testTasks[i]);
    }

    double ll = waitForFutures();

    return ll;
  }

  public double computeDocLLH(int d, TraverseMap dk) {
    double ll = 0;
    for (int j = 0; j < dk.size; j++) {
      short count = dk.value[dk.idx[j]];
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

  class Metric {

    public int num1;
    public int num2;

    public Metric() {
      num1 = 0;
      num1 = 0;
    }
  }

  class SampleTask implements Callable<Double> {

    private ConcurrentLinkedQueue<Integer> taskQueue;
    private float[] p;
    private short[] tidx;
    private FTree tree;
    private int[] nk;
    private int[] wk;
    public Metric metric;

    public SampleTask() {
      p = new float[K];
      tidx = new short[K];
      tree = new FTree(K);
      wk = new int[K];
      nk = new int[K];
    }

    public void setTaskQueue(ConcurrentLinkedQueue<Integer> taskQueue, int[] nk) {
      this.taskQueue = taskQueue;
      System.arraycopy(nk, 0, this.nk, 0, K);
    }

    @Override
    public Double call() {
      Integer wid;
      double ll = 0;
      metric = new Metric();
      while ((wid = taskQueue.poll()) != null) {
        Arrays.fill(wk, 0);
        for (int wi = mat.rs[wid]; wi < mat.rs[wid + 1]; wi ++) {
          wk[topics[wi]] ++;
        }

        buildFTree(wk, tree, p, nk);
        sample(wid, tree, p, tidx, nk, wk, metric);

        for (int k = 0; k < K; k ++)
          if (wk[k] > 0) {
            ll += Gamma.logGamma(wk[k] + beta) - lgammaBeta;
          }
      }
      return ll;
    }

    public int[] getNk() {
      return nk;
    }
  }

  class InitTask implements Callable<Double> {
    private ConcurrentLinkedQueue<Integer> taskQueue;
    public int[] nk;

    public InitTask(ConcurrentLinkedQueue<Integer> queue) {
      taskQueue = queue;
      nk = new int[K];
    }

    @Override
    public Double call() throws Exception {
      Integer wid;
      Random rand = new Random(System.currentTimeMillis());
      while ((wid = taskQueue.poll()) != null) {
        init(wid, rand, nk);
      }
      return 0.0;
    }
  }

  class TestTask implements Callable<Double> {
    private ConcurrentLinkedQueue<Integer> taskQueue;

    public TestTask() {

    }

    public void setTaskQueue(ConcurrentLinkedQueue<Integer> queue) {
      taskQueue = queue;
    }

    @Override
    public Double call() throws Exception {
      Integer d;
      double ll = 0;
      while ((d = taskQueue.poll()) != null) {
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
    float alpha = 50.0F / 1024;
    float beta  = 0.01F;
    FastFTLDA lda = new FastFTLDA(M, V, K, alpha, beta, docs, 4);
    lda.initParallel();

    for (int i = 0; i < 100; i ++) {
      lda.iteration(i);
    }
  }

  public static void cmds(String[] argv) throws IOException {
    String path = argv[0];
    List<Document> docs = Utils.read(path);
    System.out.println("finish reading " + docs.size() + " documents");
    int M = docs.size();
    int V = Integer.parseInt(argv[1]);
    int K = Integer.parseInt(argv[2]);
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    int taskNum = Integer.parseInt(argv[3]);

    System.out.format("alpha=%f beta=%f V=%d K=%d\n",
            alpha, beta, V, K);

    FastFTLDA lda = new FastFTLDA(M, V, K, alpha, beta, docs, taskNum);
    lda.initParallel();
    int I = Integer.parseInt(argv[4]);
    for (int i = 0; i < I; i ++)
      lda.iteration(i);
  }

  public static void main(String[] argv) throws IOException {
//    nips(argv);
    cmds(argv);
  }
}