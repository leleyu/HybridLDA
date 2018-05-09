package lda.jni;

import lda.utils.Document;
import lda.utils.Utils;

import java.io.IOException;
import java.util.List;

public class WarpJNI {

  public int D, V, K, N;
  public double alpha, beta, vbeta;

  int[] ws, ds, widx, nk, tk, nidx;
  int[] used;
  short[] topics, mhs;

  short[] topics_old;

  int[] nwk;

  public final int MH_STEPS = 1;

  public WarpJNI(int V, int K, double alpha, double beta,
                 String path) {
    this.D = 0;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta = beta;
    this.vbeta = V * beta;

    try {
      load(path);
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }
  }

  public void load(String path) throws IOException {
    List<Document> docs = Utils.read(path);
    D = docs.size();

    for (int d = 0; d < D; d ++) {
      Document doc = docs.get(d);
      doc.docId = d;
      N += doc.length;
    }

    alloc();

    buildMat(docs);
  }

  public void buildMat(List<Document> docs) {

    int[] wcnt = new int[V];

    // count word and build doc start idx
    ds[0] = 0;
    for (int d = 0; d < D; d ++) {
      Document doc = docs.get(d);
      for (int w = 0; w < doc.length; w ++)
        wcnt[doc.wids[w]] ++;
      ds[d + 1] = ds[d] + doc.length;
    }

    // build word start idx
    ws[0] = 0;
    for (int w = 0; w < V; w ++)
      ws[w + 1] = ws[w] + wcnt[w];


    // build doc to word reverse idx
    for (int d = D - 1; d >= 0; d --) {
      int di = ds[d];
      Document doc = docs.get(d);
      int wid;
      for (int w = 0; w < doc.length; w ++) {
        wid = doc.wids[w];
        wcnt[wid] --;
        int pos = ws[wid] + wcnt[wid];
        widx[di + w] = pos;
      }
    }
  }

  public void alloc() {
    ws = new int[V + 1];
    ds = new int[D + 1];
    widx = new int[N];
    nk = new int[K];
    tk = new int[K];
    nidx = new int[K];
    used = new int[K];
    topics = new short[N];
    topics_old = new short[N];
    mhs = new short[MH_STEPS * N];

    nwk = new int[V * K];
  }

  public void backupTopics() {
    System.arraycopy(topics, 0, topics_old, 0, N);
  }

  public void init() {
    lda.jni.Utils.warpInit(D, V, K, ws, ds, widx, nk, topics, mhs, MH_STEPS);
    initUpdate();
  }

  public void trainOneIter(int iter) {
    long start, end;
    long doc_tt, word_tt;

    start = System.currentTimeMillis();
//    lda.jni.Utils.warpOneIter(iter, D, V, K, N, ws, ds, widx, nk, tk, nidx, used,
//            topics, mhs, MH_STEPS, alpha, beta, vbeta);
    backupTopics();
    lda.jni.Utils.visitByDoc(iter, D, V, K, N, ds, widx, nk, topics, mhs, MH_STEPS, alpha, beta, vbeta);
    end = System.currentTimeMillis();

    doc_tt = end - start;

    putUpdate();

    backupTopics();

    for (int w = 0; w < V; w ++) {
      lda.jni.Utils.visitByWord(iter, D, V, K, N, w, nwk, ws, nk, topics, mhs, MH_STEPS, alpha, beta, vbeta);
    }


    System.out.format("iter=%d iter_tt=%d\n", iter, (end - start));
  }

  public void initUpdate() {

  }

  public void updateNwk() {
    for (int w = 0; w < V; w ++) {

    }
  }

  public void buildnwk() {

    short kk;
    for (int w = 0; w < V; w ++) {
      int wp = w * K;
      for (int wi = ws[w]; wi < ws[w + 1]; wi ++) {
        kk = topics[wi];
        nwk[wp + kk] ++;
      }
    }
  }

  public void putUpdate() {
    // use topics_old and topics to update ps

  }


  public static void main(String[] argv) {
    System.loadLibrary("warp_jni");

    String path = "nips.train";

    int V = 12420;
    int K = 1024;
    double alpha = 0.1;
    double beta  = 0.1;

    WarpJNI warp = new WarpJNI(V, K, alpha, beta, path);

    System.out.format("tklen=%d\n", warp.tk.length);

    warp.init();

    System.out.format("tklen=%d\n", warp.tk.length);
    for (int i = 0; i < 20; i ++)
      warp.trainOneIter(i);

  }
}