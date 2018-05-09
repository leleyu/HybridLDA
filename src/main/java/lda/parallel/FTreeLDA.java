package lda.parallel;

import lda.sparse.*;
import lda.utils.*;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * Created by leleyu on 2016/3/9.
 */
public class FTreeLDA {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public CsrMat mat;
  public short[] topics;
  public int[] nk;

  public int taskNum;

  TraverseHashMap[] ndk;
  public int[] docLens;

  SampleTask[] sampleTasks;
  TestTask[] testTasks;
  InitTask[] initTasks;
  Future<Double>[] futures;

  public double lgammaBeta;
  public double lgammaAlpha;
  public double lgammaAlphaSum;

  public ThreadPoolExecutor executor;

  private int step = 5;

  private boolean test = false;



  public FTreeLDA(int D, int V, int K, float alpha, float beta,
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

  public void initDk() {

    for (int d = 0; d < D; d ++) {
      if (docLens[d] < Byte.MAX_VALUE) {
        ndk[d] = new S2BTraverseMap(docLens[d]);
      } else if (docLens[d] < (K / 2)) {
        ndk[d] = new S2STraverseMap(Math.min(K, docLens[d]));
      } else {
        ndk[d] = new S2STraverseArray(K);
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

  public void buildFTree(int[] wk, FTree tree, float[] p, int[] nk) {
    for (int topic = 0; topic < K; topic ++) {
      p[topic] = (wk[topic] + beta) / (nk[topic] + vbeta);
    }

    tree.build(p);
  }

  public float build(S2STraverseMap map, float[] p, short[] tidx, FTree tree) {
    float psum = 0.0F;
    short topic;
    short count;
    int idx = 0;

    for (int i = 0; i < map.size; i++) {
      topic = map.key[map.idx[i]];
      count = map.value[map.idx[i]];
      psum += count * tree.get(topic);
      p[idx] = psum;
      tidx[idx ++] = topic;
    }

    return psum;
  }

  public float build(S2BTraverseMap map, float[] p, short[] tidx, FTree tree) {
    float psum = 0.0F;
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

  public float build(S2STraverseArray map, float[] p, short[] tidx, FTree tree) {
    float psum = 0.0F;
    short topic;
    short count;
    int idx = 0;

    for (int i = 0; i < map.size; i ++) {
      topic = map.idx[i];
      count = map.value[topic];
      psum += count * tree.get(topic);
      p[idx] = psum;
      tidx[idx++] = topic;
    }

    return psum;
  }

  public float buildDocDist(int did, float[] p, short[] tidx, FTree tree) {
    TraverseHashMap map = ndk[did];

    if (map instanceof S2STraverseMap)
      return build((S2STraverseMap) map, p, tidx, tree);
    if (map instanceof S2BTraverseMap)
      return build((S2BTraverseMap) map, p, tidx, tree);
    if (map instanceof S2STraverseArray)
      return build((S2STraverseArray) map, p, tidx, tree);

    System.out.println("Error");
    return 0;
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

  public long sample(int wid, FTree tree, float[] p, short[] tidx, int[] nk, int[] wk, Random random) {
//    Random random = new Random(System.currentTimeMillis());

    float value;
    int si, ei;
    si = mat.rs[wid];
    ei = mat.rs[wid + 1];
    int[] tokens = mat.cols;
    int d, size, idx;
    short kk;
    float psum, u;
    TraverseHashMap dk;
    long K_d = 0;

    for (int wi = si; wi < ei; wi ++) {
      d = tokens[wi];
      dk = ndk[d];
      kk = topics[wi];

      wk[kk] --;
      nk[kk] --;
      value = (wk[kk] + beta) / (nk[kk] + vbeta);
      tree.update(kk, value);

      synchronized (dk) {
        dk.dec(kk);
        size = dk.size;
        K_d += size;
        psum = buildDocDist(d, p, tidx, tree);
//        psum = 0.0F;
//        for (int i = 0; i < dk.size; i ++) {
//          short topic = dk.key[dk.idx[i]];
//          short count = dk.value[dk.idx[i]];
//          psum += count * tree.get(topic);
//          p[i] = psum;
//          tidx[i] = topic;
//        }

        u = random.nextFloat() * (psum + alpha * tree.first());

        if (u < psum) {
          u = random.nextFloat() * psum;
          idx = BinarySearch.binarySearch(p, u, 0, size - 1);
          kk = tidx[idx];
        } else {
          kk = (short) tree.sample(random.nextFloat() * tree.first());
        }

        dk.inc(kk);
      }

      wk[kk] ++;
      nk[kk] ++;
      value = (wk[kk] + beta) / (nk[kk] + vbeta);
      tree.update(kk, value);
      topics[wi] = kk;
    }

    return K_d;
  }

  public double trainParallel(int it) {

    AtomicInteger wids = new AtomicInteger(0);

    for (int t = 0; t < taskNum; t ++) {
      sampleTasks[t].setInteger(wids, nk);
      futures[t] = executor.submit(sampleTasks[t]);
    }

    double ll = waitForFutures();
    reduceNk();
    long K_d_sum = 0;
    for (int t = 0; t < taskNum; t ++)
      K_d_sum += sampleTasks[t].K_d;

    double K_d = K_d_sum * 1.0 / N;

    System.out.format("iteration=%d k_d_sum=%d k_d=%f\n", it, K_d_sum, K_d);

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

    if (it % 5 == 0)
      test = true;
    else
      test = false;
    long start;
    long train_tt, eval_tt;
    start = System.currentTimeMillis();
    double ll = trainParallel(it);
    train_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    if (test)
      ll += loglikelihood();

    eval_tt = System.currentTimeMillis() - start;
    System.out.format("it=%d ll=%f train_tt=%d eval_tt=%d\n",
            it, ll, train_tt, eval_tt);
  }

  public double computeDocLLH() {

    AtomicInteger dids = new AtomicInteger(0);

    for (int i = 0; i < taskNum; i ++) {
      testTasks[i].setDids(dids);
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

  class SampleTask implements Callable<Double> {

    private ConcurrentLinkedQueue<Integer> taskQueue;
    private float[] p;
    private short[] tidx;
    private FTree tree;
    private int[] nk;
    private int[] wk;
    private long K_d;

    private AtomicInteger wids;

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
      K_d = 0;
    }

    public void setInteger(AtomicInteger wids, int[] nk) {
      this.wids = wids;
      System.arraycopy(nk, 0, this.nk, 0, K);
      K_d = 0;
    }

    @Override
    public Double call() {
      double ll = 0;
      Random rand = new Random(System.currentTimeMillis());

      while (true) {
        int wid = wids.getAndIncrement();
        if (wid >= V)
          break;

        Arrays.fill(wk, 0);
        for (int wi = mat.rs[wid]; wi < mat.rs[wid + 1]; wi++) {
          wk[topics[wi]]++;
        }

        buildFTree(wk, tree, p, nk);
        K_d += sample(wid, tree, p, tidx, nk, wk, rand);

        if (test) {
          for (int k = 0; k < K; k++)
            if (wk[k] > 0) {
              ll += Gamma.logGamma(wk[k] + beta) - lgammaBeta;
            }
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
    private AtomicInteger dids;

    public TestTask() {

    }

    public void setTaskQueue(ConcurrentLinkedQueue<Integer> queue) {
      taskQueue = queue;
    }

    public void setDids(AtomicInteger dids) {
      this.dids = dids;
    }

    @Override
    public Double call() throws Exception {
//      Integer d;
      double ll = 0;
//      while ((d = taskQueue.poll()) != null) {
//        ll += computeDocLLH( d, ndk[d]);
//      }

      while (true) {
        int did = dids.getAndIncrement();
        if (did >= D)
          break;
        ll += computeDocLLH(did, ndk[did]);
      }
      return ll;
    }
  }

  class Sampler extends Thread {

    public AtomicInteger wids;

    private float[] p;
    private short[] tidx;
    private FTree tree;
    private int[] nk;
    private int[] wk;

    private double ll;

    public Sampler() {
      p = new float[K];
      tidx = new short[K];
      tree = new FTree(K);
      wk = new int[K];
      nk = new int[K];
    }

    @Override
    public void run() {
      ll = 0;
      Random rand = new Random(System.currentTimeMillis());

      while (true) {
        int wid = wids.getAndIncrement();
        if (wid >= V)
          break;

        Arrays.fill(wk, 0);
        for (int wi = mat.rs[wid]; wi < mat.rs[wid + 1]; wi++) {
          wk[topics[wi]]++;
        }

        buildFTree(wk, tree, p, nk);
        sample(wid, tree, p, tidx, nk, wk, rand);

        if (test) {
          for (int k = 0; k < K; k++)
            if (wk[k] > 0) {
              ll += Gamma.logGamma(wk[k] + beta) - lgammaBeta;
            }
        }

      }
    }
  }

  public static FTreeLDA read(String[] argv) throws IOException {
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

    FTreeLDA lda = new FTreeLDA(M, V, K, alpha, beta, docs, taskNum);
    return lda;
  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int M = docs.size();
    int V = 12420;
    int K = 1000;
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    FTreeLDA lda = new FTreeLDA(M, V, K, alpha, beta, docs, 4);
    lda.initDk();
    lda.initParallel();

    for (int i = 0; i < 100; i ++) {
      lda.iteration(i);
    }
  }

  public static void cmds(String[] argv) throws IOException {
    FTreeLDA lda = read(argv);

    long start = System.currentTimeMillis();
    lda.initDk();
    lda.initParallel();
    long init_tt = System.currentTimeMillis() - start;
    System.out.format("init_tt=%d\n", init_tt);
    int I = Integer.parseInt(argv[4]);
    for (int i = 0; i < I; i ++)
      lda.iteration(i);

    lda.executor.shutdown();
  }

  public static void main(String[] argv) throws IOException {
//    nips(argv);
    cmds(argv);
  }
}