package lda.parallel;

import lda.sparse.S2BTraverseMap;
import lda.sparse.S2STraverseArray;
import lda.sparse.S2STraverseMap;
import lda.sparse.TraverseHashMap;
import lda.utils.BinarySearch;
import lda.utils.Document;
import lda.utils.FTree;
import lda.utils.Utils;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by leleyu on 2016/12/26.
 */
public class CombineLDA {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public int[] nk;

  // Data structures
  public int[] ws;
  public int[] didx;
  public short[] topics;
  public short[] mhs;

  public int[] ds;
  public int[] widx;

  public int MH_STEPS = 2;
  public int maxlen;

  TraverseHashMap[] ndk;
  public int[] docLens;
  public boolean[] isShort;

  public double lgamma_beta;
  public double lgamma_alpha;

  public boolean test = false;

  Task[] initTasks;
  Task[] ftreeTasks;
  Future[] futures;

  public int taskNum;
  public ThreadPoolExecutor executor;

  public CombineLDA(int D, int V, int K, float alpha, float beta,
                        List<Document> docs, int taskNum, int length) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta  = beta;
    this.vbeta = V * beta;

    docLens = new int[D];
    isShort = new boolean[D];

    System.out.format("length=%d\n", length);
    buildMat(docs, length);
    System.out.format("N=%d MH_STEPS=%d\n", N, MH_STEPS);

    mhs = new short[N * MH_STEPS];
    nk  = new int[K];

    this.lgamma_alpha = Gamma.logGamma(alpha);
    this.lgamma_beta  = Gamma.logGamma(beta);

    initDk();

    this.taskNum = taskNum;
    initTasks();
  }

  public void buildMat(List<Document> docs, int length) {
    // partition

    docLens = new int[D];
    Document doc;

    int ftreeNum = 0, warpNum = 0;
    maxlen = 0;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      docLens[d] = doc.length;
      maxlen = Math.max(maxlen, doc.length);

      if (doc.length >= length) {
        isShort[d] = false;
        warpNum ++;
      }
      else {
        isShort[d] = true;
        ftreeNum ++;
      }
    }

    System.out.format("ftreeNum=%d warpNum=%d\n",
            ftreeNum, warpNum);

    // build
    int[] wcnt = new int[V];
    N = 0;
    ds = new int[D + 1];

    // count word and build doc start index
    ds[0] = 0;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int w = 0; w < doc.length; w ++)
        wcnt[doc.wids[w]] ++;
      ds[d + 1] = ds[d] + doc.length;
      N += doc.length;
    }

    topics = new short[N];
    didx   = new int[N];
    widx   = new int[N];
    ws     = new int[V + 1];
    // build word start idx
    ws[0] = 0;
    for (int i = 0; i < V; i ++)
      ws[i + 1] = ws[i] + wcnt[i];

    // build doc to word reverse index
    for (int d = D - 1; d >= 0; d --) {
      int di = ds[d];
      doc = docs.get(d);
      int wid;
      for (int w = 0; w < doc.length; w ++) {
        wid = doc.wids[w];
        int pos = ws[wid] + (--wcnt[wid]);
        widx[di + w] = pos;
        didx[pos] = d;
      }
    }
  }

  public void initDk() {
    ndk = new TraverseHashMap[D];

    for (int d = 0; d < D; d ++) {
      if (isShort[d]) {
        initDk(d);
      }
    }
  }

  public void initDk(int d) {
    if (docLens[d] < Byte.MAX_VALUE) {
      ndk[d] = new S2BTraverseMap(docLens[d]);
    } else if (docLens[d] < (K / 2)) {
      ndk[d] = new S2STraverseMap(Math.min(K, docLens[d]));
    } else {
      ndk[d] = new S2STraverseArray(K);
    }
  }

  public void initTasks() {
    this.initTasks = new InitTask[taskNum];
    this.ftreeTasks = new FTreeTask[taskNum];
    this.futures = new Future[taskNum];

    for (int i = 0; i < taskNum; i++) {
      initTasks[i] = new InitTask();
      ftreeTasks[i] = new FTreeTask();
    }

    executor = new ThreadPoolExecutor(16, 16, 100000,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<Runnable>());
  }

  public void buildFTree(int[] wk, FTree tree, float[] p, int[] nk) {
    for (int topic = 0; topic < K; topic ++)
      p[topic] = (wk[topic] + beta) / (nk[topic] + vbeta);

    tree.build(p);
  }

  public void init() {

    AtomicInteger wids = new AtomicInteger(0);

    for (int i = 0; i < taskNum; i ++) {
      initTasks[i].set(wids, nk);
      futures[i] = executor.submit(initTasks[i]);
    }

    waitForFutures();
    reduceNk(initTasks);
  }

  public void init(int wid, Random rand, int[] nk) {
    for (int wi = ws[wid]; wi < ws[wid + 1]; wi ++) {
      short tt = (short) rand.nextInt(K);
      topics[wi] = tt;
      nk[tt] ++;

      int did = didx[wi];
      if (isShort[did]) {
        synchronized (ndk[did]) {
          ndk[did].inc(tt);
        }
        mhs[wi * MH_STEPS] = -1;
      } else {
        for (int m = 0; m < MH_STEPS; m ++)
          mhs[wi * MH_STEPS + m] = tt;
      }
    }
  }

  public float buildDocDist(int did, float[] p, short[] tidx, FTree tree) {
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
        psum += count * tree.get(topic);
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
        psum += count * tree.get(topic);
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
        psum += count * tree.get(topic);
        p[idx] = psum;
        tidx[idx ++] = topic;
      }
      return psum;
    }

    for (int i = 0; i < map.size; i ++) {
      topic = map.getKey(i);
      count = map.getVal(i);
      psum += count * tree.get(topic);
      p[idx] = psum;
      tidx[idx ++] = topic;
    }
    return psum;
  }

  public void sample(int wid, FTree tree, float[] p, short[] tidx, int[] nk, int[] wk, Random random) {
//    Random random = new Random(System.currentTimeMillis());

    float value;
    int si, ei;
    si = ws[wid];
    ei = ws[wid + 1];
    int[] tokens = didx;
    int d, size, idx;
    short kk;
    float psum, u;
    TraverseHashMap dk;

    for (int wi = si; wi < ei; wi ++) {
      d = tokens[wi];
      if (!isShort[d])
        continue;

      dk = ndk[d];
      kk = topics[wi];

      wk[kk] --;
      nk[kk] --;
      value = (wk[kk] + beta) / (nk[kk] + vbeta);
      tree.update(kk, value);

      synchronized (dk) {
        dk.dec(kk);
        size = dk.size;
        psum = buildDocDist(d, p, tidx, tree);
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

  public void trainFTree() {
    AtomicInteger wids = new AtomicInteger(0);

    for (int i = 0; i < taskNum; i ++) {
      ftreeTasks[i].set(wids, nk);
      futures[i] = executor.submit(ftreeTasks[i]);
    }

    waitForFutures();
    reduceNk(ftreeTasks);
  }

  public double trainWarp(int it) {
    return lda.jni.Utils.warpOneIter(it, D, V, K, N,
      ws, ds, widx, nk,
      didx, isShort,
      topics, mhs,
      MH_STEPS, alpha, beta, vbeta, maxlen);
  }

  public double computeDocLLH() {
    double ll = 0;
    for (int d = 0; d < D; d ++) {
      if (isShort[d] && ndk[d] != null)
        ll += computeDocLLH(d, ndk[d]);
    }
    return ll;
  }

  public double computeDocLLH(int d, TraverseHashMap dk) {
    double ll = 0;
    for (int j = 0; j < dk.size; j++) {
      short count = dk.getVal(j);
      ll += Gamma.logGamma(alpha + count) - lgamma_alpha;
    }
    ll -= Gamma.logGamma(alpha * K + docLens[d]) - Gamma.logGamma(alpha * K);
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

  public void builddk() {
    for (int d = 0; d < D; d ++) {
      if (isShort[d] && ndk[d] == null) {
        initDk(d);
        for (int di = ds[d]; di < ds[d + 1]; di ++) {
          short tt = topics[widx[di]];
          mhs[widx[di] * MH_STEPS] = -1;
          ndk[d].inc(tt);
        }
      }
    }
  }

  public void iteration(int it) {

    test = true;
    double ll = 0;

    long start;


    start = System.currentTimeMillis();
    trainFTree();
    long ftree_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    trainWarp(it);
    ll += trainWarp(it);
    long warp_tt = System.currentTimeMillis() - start;

    if (test) {
      ll += computeDocLLH();
    }

    start = System.currentTimeMillis();
    builddk();
    long build_tt = System.currentTimeMillis() - start;

    System.out.format("it=%d ll=%f ftree_tt=%d warp_tt=%d build_tt=%d\n", it, ll,
            ftree_tt, warp_tt, build_tt);

  }

  public void waitForFutures() {
    for (int i = 0; i < taskNum; i++) {
      try {
        futures[i].get();
      } catch (InterruptedException e) {
        e.printStackTrace();
      } catch (ExecutionException e) {
        e.printStackTrace();
      }
    }
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

  abstract class Task implements Runnable {

    public int[] nk;
    public int[] tk;
    public double ll;
    protected AtomicInteger wids;

    public Task() {
      tk = new int[K];
      nk = new int[K];
    }

    public void set(AtomicInteger wids, int[] nk) {
      this.wids = wids;
      System.arraycopy(nk, 0, this.nk, 0, K);
    }

    public int[] getNk() {
      return nk;
    }
  }

  class FTreeTask extends Task {
    private float[] p;
    private short[] tidx;
    private FTree tree;

    public FTreeTask() {
      p = new float[K];
      tidx = new short[K];
      tree = new FTree(K);
    }

    @Override
    public void run() {
      ll = 0;
      Random rand = new Random(System.currentTimeMillis());
      while (true) {
        int wid = wids.getAndIncrement();
        if (wid >= V)
          break;
        Arrays.fill(tk, 0);
        for (int wi = ws[wid]; wi < ws[wid + 1]; wi ++) {
          tk[topics[wi]] ++;
        }

        buildFTree(tk, tree, p, nk);
        sample(wid, tree, p, tidx, nk, tk, rand);
      }
    }
  }

  class InitTask extends Task {

    @Override
    public void run() {
      Random rand = new Random(System.currentTimeMillis());
      while (true) {
        int wid = wids.getAndIncrement();
        if (wid >= V)
          break;
        init(wid, rand, nk);
      }
    }
  }

  public static CombineLDA read(String[] argv) throws IOException {
    String path = argv[0];
    List<Document> docs = Utils.read(path);

    System.out.println("finish reading " + docs.size() + " documents");
    int M = docs.size();
    int V = Integer.parseInt(argv[1]);
    int K = Integer.parseInt(argv[2]);
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    int taskNum = Integer.parseInt(argv[3]);
    int length  = Integer.parseInt(argv[5]);

    System.out.format("alpha=%f beta=%f V=%d K=%d\n",
            alpha, beta, V, K);

    CombineLDA lda = new CombineLDA(M, V, K, alpha, beta, docs, taskNum, length);
    return lda;
  }

  public static void cmds(String[] argv) throws IOException {
    CombineLDA lda = read(argv);

    long start = System.currentTimeMillis();
    lda.init();
    long init_tt = System.currentTimeMillis() - start;
    System.out.format("init_tt=%d\n", init_tt);
    int I = Integer.parseInt(argv[4]);
    for (int i = 0; i < I; i ++)
      lda.iteration(i);

    lda.executor.shutdown();
  }

  public static void main(String[] argv) throws IOException {
    System.loadLibrary("warp_jni");
//    nips(argv);
    cmds(argv);
  }
}
