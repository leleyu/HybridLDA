package lda.single.warp;

import lda.utils.DVMat;
import org.apache.commons.math3.special.Gamma;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by yulele on 16/9/4.
 */
public class WarpLDA {

  public int D, V, K;
  public float alpha, beta, vbeta;

  public DVMat mat;
  public int[] nk;
  public int[] tk;
  public int[] used;
  public int[] nidx;

  double ll ;

  public WarpLDA(int D, int V, int K, float alpha, float beta,
                 DVMat mat) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta  = beta;
    this.vbeta = V * beta;
    this.mat = mat;
  }

  public void init() {
    nk = new int[K];
    tk = new int[K];
    used = new int[K];
    nidx = new int[K];

    Random rand = new Random(System.currentTimeMillis());
    // word order init
    int MH_STEPS = mat.MH_STEPS;
    short[] topics = mat.topics0;
    short[] mhs = mat.mh;
    int[] ws = mat.ws;


    for (int w = 0; w < V; w ++) {
      for (int wi = ws[w]; wi < ws[w + 1]; wi ++) {
        short tt = (short) rand.nextInt(K);
        topics[wi] = tt;
        nk[tt] ++;
        for (int m = 0; m < MH_STEPS; m ++) {
          mhs[wi * MH_STEPS + m]  = tt;
        }
      }
    }
  }

  public void visitByRow() {
    short tt, ttt;
    Random rand = new Random(System.currentTimeMillis());

    short[] topics = mat.topics0;
    short[] mhs = mat.mh;
    int[] ds = mat.ds;
    int[] widx = mat.widx;

    double lgamma_alpha = Gamma.logGamma(alpha);

    for (int d = 0; d < D; d ++) {
      int N = ds[d + 1] - ds[d];

      if (N == 0)
        continue;

      int si = ds[d];
      int ei = ds[d + 1];
      int idx = 0;
      // compute dk for d on the fly
      Arrays.fill(tk, 0);
      for (int di = si; di < ei; di ++) {
        short k = topics[widx[di]];
        tk[k] ++;
        if (used[k] == 0) {
          used[k] = 1;
          nidx[idx ++] = k;
        }
      }

      for (int i = 0; i < idx; i ++) {
        ll += Gamma.logGamma(alpha + tk[nidx[i]]) - lgamma_alpha;
        used[nidx[i]] = 0;
      }

      ll -= Gamma.logGamma(alpha * K + N) - Gamma.logGamma(alpha * K);

      // traverse d first time, accept word proposal
      for (int di = si; di < ei; di ++) {
        idx = widx[di];

        tt = topics[idx];
        ttt = mhs[idx];

        if (tt != ttt) {
          nk[tt]--;
          tk[tt]--;

          float p = ((tk[ttt] + alpha) * (nk[tt] + vbeta)) / ((tk[tt] + alpha) * (nk[ttt] + vbeta));
          if (rand.nextFloat() < p) {
            tt = ttt;
          }

          nk[tt]++;
          tk[tt]++;
          topics[idx] = tt;
        }
      }

      // traverse d second time, generate doc proposal
      float p = (alpha * K) / (alpha * K + N);
      for (int di = si; di < ei; di++) {
        if (rand.nextFloat() < p) {
          mhs[widx[di]] = (short) rand.nextInt(K);
        } else {
          mhs[widx[di]] = topics[widx[ds[d] + rand.nextInt(N)]];
        }
      }
    }
  }

  public void visitByCol() {
    short tt, ttt;

    Random rand = new Random(System.currentTimeMillis());

    short[] topics = mat.topics0;
    short[] mhs = mat.mh;
    int[] ws = mat.ws;

    double lgamma_beta = Gamma.logGamma(beta);

    for (int w = 0; w < V; w ++) {

      int len = ws[w + 1] - ws[w];

      if (len == 0)
        continue;

      // compute wk for w on the fly
      Arrays.fill(tk, 0);
      int idx = 0;
      for (int wi = ws[w]; wi < ws[w + 1]; wi++) {
        short k = topics[wi];
        tk[k]++;
        if (used[k] == 0) {
          used[k] = 1;
          nidx[idx ++] = k;
        }
      }

      for (int i = 0; i < idx; i ++) {
        ll += Gamma.logGamma(beta + tk[nidx[i]]) - lgamma_beta;
        used[nidx[i]] = 0;
      }

      // traverse w first time, accept doc proposal
      int si = ws[w];
      int ei = ws[w + 1];

      for (int wi = si; wi < ei; wi++) {
        tt = topics[wi];
        ttt = mhs[wi];

        if (tt != ttt) {

          tk[tt]--;
          nk[tt]--;

          float b = tk[tt] + beta;
          float d = nk[tt] + vbeta;
          float a = tk[ttt] + beta;
          float c = nk[ttt] + vbeta;

          if (rand.nextFloat() < (a * d) / (b * c)) {
            tt = ttt;
          }

          topics[wi] = tt;
          tk[tt]++;
          nk[tt]++;
        }
      }

      // traverse two time, compute word proposal
      float p = (K * beta) / (K * beta + len);

      for (int wi = si; wi < ei; wi ++) {
        float u = rand.nextFloat();
        if (u < p) {
          mhs[wi] = (short) rand.nextInt(K);
        } else {
          idx = rand.nextInt(len);
          mhs[wi] = topics[ws[w] + idx];
        }
      }
    }

    for (int k = 0; k < K; k ++)
      ll += -Gamma.logGamma(nk[k] + vbeta) + Gamma.logGamma(vbeta);
  }

  public void trainOneIteration(int iter) {
    ll = 0;

    long start, end;
    start = System.currentTimeMillis();
    visitByCol();
    end = System.currentTimeMillis();
    long col_tt = end - start;

    start = System.currentTimeMillis();
    visitByRow();
    end = System.currentTimeMillis();
    long row_tt = end - start;

    System.out.format("iter=%d row_tt=%d col_tt=%d ll=%f\n",
            iter, row_tt, col_tt, ll);
  }


  public static void nips(String[] argv) {
    String path = "nips.train";
    int V = 12420;
    int K = 1024;
    DVMat mat = new DVMat(path, V);
    float alpha = 0.1F;
    float beta  = 0.1F;

    WarpLDA lda = new WarpLDA(mat.D, mat.V, K, alpha, beta, mat);
    long start = System.currentTimeMillis();
    lda.init();
    long end = System.currentTimeMillis();

    System.out.format("init_tt=%d\n", end - start);

    for (int i = 0; i < 50; i ++) {
      lda.trainOneIteration(i);
    }
  }

  public static void nytimes(String[] argv) {
    String path = "nytimes.train";
    int V = 102661;
    int K = 1024;
    DVMat mat = new DVMat(path, V);
    float alpha = 50.0F / K;
    float beta  = 0.01F;

    WarpLDA lda = new WarpLDA(mat.D, mat.V, K, alpha, beta, mat);
    long start = System.currentTimeMillis();
    lda.init();
    long end = System.currentTimeMillis();

    System.out.format("init_tt=%d\n", end - start);

    for (int i = 0; i < 200; i ++) {
      lda.trainOneIteration(i);
    }
  }

  public static void main(String[] argv) {
    nips(argv);
  }
}
