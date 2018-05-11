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
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by leleyu on 2016/12/25.
 */
public class CombineLDADynamic {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public int[] nk;

  // structures for long docs
  public int[] lws;
  public short[] ltopics;
  public short[] mhs;

  public int[] lds;
  public int[] lwidx;

  public int ln;
  public int ld;

  public int MAX_MH;
  public int[] mh_steps;

  // structures for short docs
  public int[] sws;
  public short[] stopics;
  public int[] sdids;

  public int sn;
  public int sd;

  TraverseHashMap[] ndk;
  public int[] docLens;
  public boolean[] isShort;

  public double lgamma_beta;
  public double lgamma_alpha;

  public int threshold = 200;
  public int maxlen = 0;

  private boolean test = false;

  Task[] initTasks;
  Task[] ftreeTasks;
  Future[] futures;

  public int taskNum;
  public ThreadPoolExecutor executor;

  public CombineLDADynamic(int D, int V, int K, float alpha, float beta,
                            List<Document> docs, int taskNum, int threshold,
                            int MAX_MH) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta  = beta;
    this.vbeta = V * beta;

    docLens = new int[D];
    isShort = new boolean[D];

    this.threshold = threshold;

    this.MAX_MH = MAX_MH;
    this.mh_steps = new int[2];
    mh_steps[0] = 2;
    mh_steps[1] = 2;
    buildMat(docs);

    System.out.format("mhs length=%d\n", ln * MAX_MH);
    mhs = new short[ln * MAX_MH];

    this.lgamma_alpha = Gamma.logGamma(alpha);
    this.lgamma_beta  = Gamma.logGamma(beta);
    this.taskNum = taskNum;

    initTasks();

    initDk();
    nk = new int[K];
  }

  public void buildMat(List<Document> docs) {
    Map.Entry<List<Document>, List<Document>> entry = partition(docs);
    List<Document> longDocs = entry.getKey();
    List<Document> shortDocs = entry.getValue();
    buildMatForLongDocs(longDocs);
    buildMatForShortDocs(shortDocs);

    System.out.format("ftreeNum=%d ftreeTokens=%d WarpNum=%d WarpTokens=%d MH=%d threshold=%d\n",
      shortDocs.size(), sn, longDocs.size(), ln, MAX_MH, threshold);
  }

  public Map.Entry<List<Document>, List<Document>> partition(List<Document> docs) {
    for (int d = 0; d < D; d ++)
      docs.get(d).docId = d;

    List<Document> shortDocs = new ArrayList<>();
    List<Document> longDocs  = new ArrayList<>();
    // partition
    Document doc;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      docLens[d] = doc.length;
      maxlen = Math.max(docLens[d], maxlen);
      doc.docId = d;
      if (doc.length >= threshold) {
        longDocs.add(doc);
        isShort[d] = false;
      }
      else {
        shortDocs.add(doc);
        isShort[d] = true;
      }
    }

    return new AbstractMap.SimpleEntry<>(longDocs, shortDocs);
  }

  public void buildMatForLongDocs(List<Document> docs) {
    Document doc;
    int[] wcnt = new int[V];

    ld = docs.size();
    ln = 0;
    lds = new int[ld + 1];

    // count word and build doc start idx
    lds[0] = 0;
    for (int d = 0; d < ld; d ++) {
      doc = docs.get(d);
//      int did = doc.docId;
      int did = d;
      for (int w = 0; w < doc.length; w ++)
        wcnt[doc.wids[w]] ++;
      lds[did + 1] = doc.length;
      ln += doc.length;
    }

    ltopics = new short[ln];
    lwidx   = new int[ln];

    // build doc start index
    for (int d = 1; d <= ld; d ++)
      lds[d] += lds[d - 1];

    // build word start idx
    lws = new int[V + 1];
    lws[0] = 0;
    for (int i = 0; i < V; i ++)
      lws[i + 1] = lws[i] + wcnt[i];

    // build doc to word reverse idx
    for (int d = ld - 1; d >= 0; d --) {
      doc = docs.get(d);
//      int did = doc.docId;
      int did = d;
      int di = lds[did];
      int wid ;
      for (int w = 0; w < doc.length; w ++) {
        wid = doc.wids[w];
        int pos = lws[wid] + (--wcnt[wid]);
        lwidx[di + w] = pos;
      }
    }
  }

  public void buildMatForShortDocs(List<Document> docs) {
    Document doc;
    int[] wcnt = new int[V];
    sws = new int[V + 1];

    sn = 0;
    sd = docs.size();

    // count word
    for (int d = 0; d < sd; d ++) {
      doc = docs.get(d);
      for (int w = 0; w < doc.wids.length; w ++)
        wcnt[doc.wids[w]] ++;
      sn += doc.wids.length;
    }

    stopics = new short[sn];
    sdids   = new int[sn];

    // build word start idx
    sws[0] = 0;
    for (int i = 0; i < V; i ++)
      sws[i + 1] = sws[i] + wcnt[i];

    // build did idx
    for (int d = sd - 1; d >= 0; d --) {
      doc = docs.get(d);
      int did = doc.docId;
      int wid;
      for (int w = 0; w < doc.length; w ++) {
        wid = doc.wids[w];
        int pos = sws[wid] + (--wcnt[wid]);
        sdids[pos] = did;
      }
    }
  }

  public void initDk() {
    ndk = new TraverseHashMap[D];
    for (int d = 0; d < D; d ++) {
      if (isShort[d]) {
        if (docLens[d] < Byte.MAX_VALUE) {
          ndk[d] = new S2BTraverseMap(docLens[d]);
        } else if (docLens[d] < (K / 2)) {
          ndk[d] = new S2STraverseMap(Math.min(K, docLens[d]));
        } else {
          ndk[d] = new S2STraverseArray(K);
        }
      }
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


  public void init(int wid, Random rand, int[] nk) {
    short tt;
    // short
    for (int wi = sws[wid]; wi < sws[wid + 1]; wi ++) {
      tt = (short) rand.nextInt(K);
      stopics[wi] = tt;
      nk[tt] ++;
      int d = sdids[wi];
      synchronized (ndk[d]) {
        ndk[d].inc(tt);
      }
    }

    // long
    for (int wi = lws[wid]; wi < lws[wid + 1]; wi ++) {
      tt = (short) rand.nextInt(K);
      ltopics[wi] = tt;
      nk[tt] ++;
      for (int m = 0; m < mh_steps[0]; m ++)
        mhs[wi * MAX_MH + m] = tt;
    }
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


  public void buildFTree(int[] wk, FTree tree, float[] p, int[] nk) {
    for (int topic = 0; topic < K; topic ++)
      p[topic] = (wk[topic] + beta) / (nk[topic] + vbeta);

    tree.build(p);
  }
  public void sample(int wid, FTree tree, float[] p, short[] tidx, int[] nk, int[] wk, Random random) {
//    Random random = new Random(System.currentTimeMillis());

    float value;
    int si, ei;
    si = sws[wid];
    ei = sws[wid + 1];
    int[] tokens = sdids;
    int d, size, idx;
    short kk;
    float psum, u;
    TraverseHashMap dk;

    for (int wi = si; wi < ei; wi ++) {
      d = tokens[wi];
      if (!isShort[d])
        continue;

      dk = ndk[d];
      kk = stopics[wi];

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
      stopics[wi] = kk;
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


  public void trainShort(int it) {
    AtomicInteger wids = new AtomicInteger(0);

    for (int i = 0; i < taskNum; i ++) {
      ftreeTasks[i].set(wids, nk);
      futures[i] = executor.submit(ftreeTasks[i]);
    }

    waitForFutures();
    reduceNk(ftreeTasks);
  }

  public double trainLong(int it) {
    return lda.jni.Utils.warpOneIterDyn(it, D, V, K, N,
      ld, ln, sd, sn,
      lws, lds, lwidx, nk,
      sws, stopics, ltopics, isShort,
      mhs, mh_steps, alpha, beta, vbeta, maxlen, MAX_MH);

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

  public void iteration(int it) {

    test = true;

    double ll = 0;

    long start;


    start = System.currentTimeMillis();
    trainShort(it);
    long ftree_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    ll += trainLong(it);
    long warp_tt = System.currentTimeMillis() - start;

    System.out.format("mh1=%d mh2=%d\n", mh_steps[0], mh_steps[1]);

    if (test) {
      ll += computeDocLLH();
    }

    System.out.format("it=%d ll=%f ftree_tt=%d warp_tt=%d\n", it, ll,
      ftree_tt, warp_tt);

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

  public double computeDocLLH() {
    double ll = 0;
    for (int d = 0; d < D; d ++) {
      if (isShort[d] && ndk[d] != null)
        ll += computeDocLLH(d, ndk[d]);
    }
    return ll;
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
        for (int wi = sws[wid]; wi < sws[wid + 1]; wi ++) {
          tk[stopics[wi]] ++;
        }

        for (int wi = lws[wid]; wi < lws[wid + 1]; wi ++) {
          tk[ltopics[wi]] ++;
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

  public static CombineLDADynamic read(String[] argv) throws IOException {
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
    int mh_steps = 10;

    System.out.format("alpha=%f beta=%f V=%d K=%d mh=%d\n",
      alpha, beta, V, K, mh_steps);

    CombineLDADynamic lda = new CombineLDADynamic(M, V, K, alpha, beta, docs, taskNum, length, mh_steps);
    return lda;
  }

  public static void cmds(String[] argv) throws IOException {
    CombineLDADynamic lda = read(argv);

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
    cmds(argv);
  }




}
