package lda.utils;

import java.io.IOException;
import java.util.List;

/**
 * Created by yulele on 16/9/4.
 */
public class DVMat {

  public int[] ws;

  public short[] topics0;
  public short[] mh;

  public int[] ds;
  public int[] widx;

  public int D, V;
  public int N;

  public int MH_STEPS;

  public DVMat(String path, int V) {
    this(path, V, 1);
  }

  public DVMat(String path, int V, int MH_STEPS) {
    this.MH_STEPS = MH_STEPS;
    this.V = V;
    this.D = 0;
    this.N = 0;

    try {
      load(path);
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }
  }

  public void alloc() {
    this.ws = new int[V + 1];
    this.ds = new int[D + 1];
    this.widx = new int[N];

    this.topics0 = new short[N];
    this.mh = new short[N * MH_STEPS];
  }

  public void load(String path) throws IOException {
    List<Document> docs = Utils.read(path);

    N = 0;
    D = docs.size();
    Document doc;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      N += doc.length;
    }

    alloc();

    buildMat(docs);

//    check(docs);
  }

  public void buildMat(List<Document> docs) {
    Document doc;
    int[] wcnt = new int[V];

    // count word and build doc start idx
    ds[0] = 0;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int w = 0; w < doc.wids.length; w ++)
        wcnt[doc.wids[w]] ++;
      ds[d + 1] = ds[d] + doc.length;
    }

    // build word start idx
    ws[0] = 0;
    for (int i = 0; i < V; i ++) {
      ws[i + 1] = ws[i] + wcnt[i];
    }

    // build doc to word reverse idx
    for (int d = D - 1; d >= 0; d --) {
      int di = ds[d];
      doc = docs.get(d);
      int wid;
      for (int w = 0; w < doc.wids.length; w ++) {
        wid = doc.wids[w];
        int pos = ws[wid] + (--wcnt[wid]);
        widx[di + w] = pos;
      }
    }
  }

  public void check(List<Document> docs) {
    int[] wridx = new int[N + 1];
    for (int w = 0; w < V; w ++) {
      for (int wi = ws[w]; wi < ws[w+1]; wi ++) {
        wridx[wi] = w;
      }
    }

    for (int d = 0; d < D; d ++) {
      Document doc = docs.get(d);
      int di = ds[d];
      for (int w = 0; w < doc.wids.length; w ++) {
        int pos = widx[di + w];
        if (doc.wids[w] != wridx[pos]) {
          System.out.println("Error");
        } else {
          System.out.format("doc.wids[w]=%d wridx[pos]=%d\n", doc.wids[w], wridx[pos]);
        }
      }
    }
  }

  public static void main(String[] argv) throws IOException {
    String path = "nips.train";

    int V = 12420;
    DVMat mat = new DVMat(path, V);
  }
}
