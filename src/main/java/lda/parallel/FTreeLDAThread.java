package lda.parallel;

import lda.sparse.S2STraverseMap;
import lda.sparse.TraverseHashMap;
import lda.utils.*;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * Created by leleyu on 2016/3/9.
 */
public class FTreeLDAThread {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public CsrMat mat;
  public short[] topics;
  public int[] nk;

  public int taskNum;

  S2STraverseMap[] ndk;
  public int[] docLens;

  public double lgammaBeta;
  public double lgammaAlpha;
  public double lgammaAlphaSum;

  private boolean test = false;

  private Sampler[] samplers;
  private TestTask[] testTasks;
  private InitTask[] initTasks;

  private float[][] p;
  private short[][] tidx;
  private int[][] tk;
  private int[][] wk;
  private FTree[] trees;

  public ThreadPoolExecutor executor;
  private Future[] futures;


  public FTreeLDAThread(int D, int V, int K, float alpha, float beta,
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
    ndk = new S2STraverseMap[D];

    this.taskNum = taskNum;

    samplers = new Sampler[taskNum];
    testTasks = new TestTask[taskNum];
    initTasks = new InitTask[taskNum];

    p = new float[taskNum][K];
    tidx = new short[taskNum][K];
    tk = new int[taskNum][K];
    wk = new int[taskNum][K];
    trees = new FTree[taskNum];

    for (int i = 0; i < taskNum; i ++) {
      p[i] = new float[K];
      tidx[i] = new short[K];
      tk[i] = new int[K];
      wk[i] = new int[K];
      trees[i] = new FTree(K);
      samplers[i] = new Sampler(i);
      testTasks[i] = new TestTask();
    }

    lgammaBeta = Gamma.logGamma(beta);
    lgammaAlpha = Gamma.logGamma(alpha);
    lgammaAlphaSum = Gamma.logGamma(alpha * K);

    executor = new ThreadPoolExecutor(taskNum, taskNum, 1000,
      TimeUnit.SECONDS,
      new LinkedBlockingQueue<Runnable>());

    futures = new Future[taskNum];
  }

  public void initDk() {
    
    for (int d = 0; d < D; d ++) {
      ndk[d] = new S2STraverseMap(Math.min(K, docLens[d]));
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

  public void initParallel() {
    AtomicInteger wids = new AtomicInteger(0);

    for (int i = 0; i < taskNum; i ++) {
      initTasks[i] = new InitTask(wids, i);
      futures[i] = executor.submit(initTasks[i]);
    }

    waitForFutures();
    reduceNk();
  }

  public void sample(int wid, FTree tree, float[] p, short[] tidx, int[] nk, int[] wk, Random random) {
//    Random random = new Random(System.currentTimeMillis());

    float value;
    int si, ei;
    si = mat.rs[wid];
    ei = mat.rs[wid + 1];
    int[] tokens = mat.cols;
    int d, size, idx;
    short kk;
    float psum, u;
    S2STraverseMap dk;

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
//        psum = build(ndk[d], p, tidx, tree);
        psum = 0.0F;
        for (int i = 0; i < dk.size; i ++) {
          short topic = dk.key[dk.idx[i]];
          short count = dk.value[dk.idx[i]];
          psum += count * tree.get(topic);
          p[i] = psum;
          tidx[i] = topic;
        }

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
  }

  public double trainParallel(int it) {

    AtomicInteger wids = new AtomicInteger(0);

    for (int t = 0; t < taskNum; t ++) {
      samplers[t].set(wids, nk);
      futures[t] = executor.submit(samplers[t]);
    }

    waitForFutures();

    double ll = 0;
    for (int t = 0; t < taskNum; t ++)
      ll += samplers[t].ll;

    reduceNk();
    return ll;
  }

  public void waitForFutures() {
    try {
      for (int i = 0; i < taskNum; i++)
        futures[i].get();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void reduceNk() {
    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = tk[i];
      for (int k = 0; k < K; k ++) {
        localNk[k] -= nk[k];
      }
    }

    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = tk[i];
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

    waitForFutures();

    double ll = 0;
    for (int i = 0; i < taskNum; i ++)
      ll += testTasks[i].ll;

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

  class InitTask implements Runnable {
    private AtomicInteger wids;
    private int taskId;

    public InitTask(AtomicInteger wids, int taskId) {
      this.wids = wids;
      this.taskId = taskId;
    }

    @Override
    public void run() {
      Random rand = new Random(System.currentTimeMillis());
      while (true) {
        int wid = wids.getAndIncrement();
        if (wid >= V)
          break;
        init(wid, rand, tk[taskId]);
      }
    }
  }

  class TestTask implements Runnable {
    private AtomicInteger dids;
    public double ll;

    public void setDids(AtomicInteger dids) {
      this.dids = dids;
    }

    @Override
    public void run() {
      ll = 0;

      while (true) {
        int did = dids.getAndIncrement();
        if (did >= D)
          break;
        ll += computeDocLLH(did, ndk[did]);
      }
    }
  }

  class Sampler implements Runnable {

    public AtomicInteger wids;
    public double ll;
    public int taskId;

    public Sampler(int taskId) {
      this.taskId = taskId;
    }

    public void set(AtomicInteger wids, int[] nk) {
      this.wids = wids;
      System.arraycopy(nk, 0, tk[taskId], 0, K);
    }

    @Override
    public void run() {
      ll = 0;
      Random rand = new Random(System.currentTimeMillis());

      while (true) {
        int wid = wids.getAndIncrement();
        if (wid >= V)
          break;

        Arrays.fill(wk[taskId], 0);
        for (int wi = mat.rs[wid]; wi < mat.rs[wid + 1]; wi++) {
          wk[taskId][topics[wi]]++;
        }

        buildFTree(wk[taskId], trees[taskId], p[taskId], tk[taskId]);
        sample(wid, trees[taskId], p[taskId], tidx[taskId], tk[taskId], wk[taskId], rand);

        if (test) {
          for (int k = 0; k < K; k++)
            if (wk[taskId][k] > 0) {
              ll += Gamma.logGamma(wk[taskId][k] + beta) - lgammaBeta;
            }
        }

      }
    }
  }

  public static FTreeLDAThread read(String[] argv) throws IOException {
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

    FTreeLDAThread lda = new FTreeLDAThread(M, V, K, alpha, beta, docs, taskNum);
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
    FTreeLDAThread lda = new FTreeLDAThread(M, V, K, alpha, beta, docs, 4);
    lda.initDk();
    lda.initParallel();

    for (int i = 0; i < 100; i ++) {
      lda.iteration(i);
    }
  }

  public static void cmds(String[] argv) throws IOException {
    FTreeLDAThread lda = read(argv);

    long start = System.currentTimeMillis();
    lda.initDk();
    lda.initParallel();
    long init_tt = System.currentTimeMillis() - start;
    System.out.format("init_tt=%d\n", init_tt);
    int I = Integer.parseInt(argv[4]);
    for (int i = 0; i < I; i ++)
      lda.iteration(i);
  }

  public static void main(String[] argv) throws IOException {
//    nips(argv);
    cmds(argv);
  }
}