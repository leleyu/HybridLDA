package lda.single.doc;

import lda.parallel.Alias;
import lda.utils.*;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by leleyu on 2016/1/27.
 */
class AliasLDA {

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

  public AliasLDA(int D, int V, int K, float alpha, float beta, List<Document> docs) {

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

    Random rand = new Random(System.currentTimeMillis());
    for (int d = 0; d < D; d ++)
      init(d, rand, nk);
  }

  public void init(int d, Random rand, int[] nk) {
    for (int j = ds[d]; j < ds[d + 1]; j ++) {
      short tt = (short) rand.nextInt(K);
      topics[j] = tt;
      nk[tt] ++;
      int wid = wids[j];
      nwk[wid][tt] ++;
    }
  }

  private void build() {
    for (int wid = 0; wid < V; wid ++) {
      build(wid, alias[wid], nwk[wid]);
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

  public double trainOneDoc(int d, int[] dk, short[] didx, short[] poss) {
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

      nwk[wid][tt] --;
      nk[tt] --;
      dk[tt] --;

      if (dk[tt] == 0) {
        size --;
        didx[poss[tt]] = didx[size];
        poss[didx[size]] = poss[tt];
      }

      // Compute pdw
      float psum = 0.0F;

      for (int i = 0; i < size; i ++) {
        kk = didx[i];
        vv = dk[kk];
        psum += vv * (nwk[wid][kk] + beta) / (nk[kk] + vbeta);
        p[i] = psum;
      }

      // probability to select pdw
      float select_pr = psum / (psum + alpha * qws[wid]);

      // MHV to draw a new topic
      for (int r = 0; r < MH_STEPS; r ++) {
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


      topics[j] = tt;
      nwk[wid][tt] ++;

      dk[tt] ++;
      if (dk[tt] == 1) {
        didx[size] = tt;
        poss[tt] = size;
        size ++;
      }
      nk[tt] ++;
    }

    double ll = 0;
    int len = ds[d + 1] - ds[d];
    ll += lgammaAlphaSum - Gamma.logGamma(alpha * K + len);
    for (int i = 0; i < K; i++) {
      if (dk[i] != 0)
        ll += Gamma.logGamma(alpha + dk[i]) - lgammaAlpha;
    }

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

  public double computeWordLLH() {
    double ll = 0;
    for (int w = 0; w < V; w ++) {
      ll += computeWordLLH(w);
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

  public void trainOneIteration(int iter) {
    long start, end;

    start = System.currentTimeMillis();
    build();
    end   = System.currentTimeMillis();

    long build_tt = end - start;

    short[] didx = new short[K];
    short[] poss = new short[K];
    int[] dk     = new int[K];

    double ll = 0;

    start = System.currentTimeMillis();
    for (int d = 0; d < D; d ++)
      ll += trainOneDoc(d, dk, didx, poss);

    end = System.currentTimeMillis();
    long train_tt = end - start;

    start = System.currentTimeMillis();
    ll += computeWordLLHSummary();
    ll += computeWordLLH();
    end = System.currentTimeMillis();
    long eval_tt = end - start;

    System.out.format("iter=%d train_tt=%d build_tt=%d eval_tt=%d llhw=%f\n",
            iter, train_tt, build_tt, eval_tt, ll);
  }

  public void testOneIteration() {

  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int D = docs.size();
    int V = 12420;
    int K = 1024;
    float alpha = 50.0F / K;
    float beta  = 0.01F;

    AliasLDA lda = new AliasLDA(D, V, K, alpha, beta, docs);
    lda.init();

    for (int i = 0; i < 200; i ++) {
      lda.trainOneIteration(i);
    }
  }

  public static void main(String [] argv) throws IOException {
    nips(argv);
  }
}