package lda.single.word;

import lda.sparse.S2SSparseMap;
import lda.utils.*;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by leleyu on 2016/9/4.
 */
public class FTreeLDA implements WordLDA {

  public int D, V, K;
  public float alpha, beta, vbeta;

  public int[] nk;
  public S2SSparseMap[] ndk;

  public List<Document> docs;
  public TokensAndParams[] words;
  float[] p;
  int[] tidx;

  public FTreeLDA(int D, int V, int K, float alpha, float beta,
                  List<Document> docs) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta = beta;
    this.vbeta = V * beta;
    this.docs = docs;
  }

  @Override
  public void init() {
    this.nk  = new int[K];
    this.p   = new float[K];
    this.tidx= new int[K];
    this.ndk = new S2SSparseMap[D];

    for (int d = 0; d < D; d ++) {
      ndk[d] = new S2SSparseMap(Math.min(K, docs.get(d).length));
    }

    this.words = Utils.buildTokensByWord(docs, K, V);

    int did, tt;
    Random rand = new Random(System.currentTimeMillis());
    for (int w = 0; w < V; w ++) {
      TokensAndParams param = words[w];
      if (param == null)
        continue;

      for (int d = 0; d < param.tokens.length; d ++) {
        did = param.tokens[d];
        tt  = rand.nextInt(K);

        param.wtrow[tt] ++;
        nk[tt] ++;
        ndk[did].inc(tt);
        param.topics[d] = (short) tt;
      }
    }
  }

  @Override
  public void trainOneIteration(int iter) {
    long start, end;

    FTree tree = new FTree(K);

    start = System.currentTimeMillis();
    for (TokensAndParams param: words)
      if (param != null)
        sampleOneWord(param, tree);

    end = System.currentTimeMillis();

    long train_tt = end - start;

    start = System.currentTimeMillis();
    double llhw = new Utils().loglilikehood2(alpha, beta, words, ndk, nk, K, V, docs);
    end = System.currentTimeMillis();

    long eval_tt = end - start;

    System.out.println(String.format("iter=%d train_tt=%d eval_tt=%d llhw=%f",
            iter, train_tt, eval_tt, llhw));
  }

  public void buildFTree(int[] wa, FTree tree) {
    for (int k = 0; k < K; k ++)
      p[k] = (wa[k] + beta) / (nk[k] + vbeta);

    tree.build(p);
  }

  public void sample(TokensAndParams param, FTree tree, int[] wa, int d, Random rand,
                     S2SSparseMap dk) {
    int tt, ttt;
    int kk, vv;
    float psum, u;

    tt  = param.topics[d];
    wa[tt] --;
    nk[tt] --;
    dk.dec(tt);
    tree.update(tt, (wa[tt] + beta) / (nk[tt] + vbeta));

    psum = 0.0F;

    int idx = 0;
    for (int i = 0; i < dk.n; i ++) {
      if (dk.key[i] != -1) {
        kk = dk.key[i];
        vv = dk.value[i];
        psum += vv * tree.get(kk);
        p[idx] = psum;
        tidx[idx ++] = kk;
      }
    }

    u = rand.nextFloat() * (psum + alpha * tree.first());
    if (u < psum) {
      u = rand.nextFloat() * psum;
      ttt = tidx[BinarySearch.binarySearch(p, u, 0, idx - 1)];
    } else {
      ttt = tree.sample(rand.nextFloat() * tree.first());
    }

    wa[ttt] ++;
    nk[ttt] ++;
    dk.inc(ttt);
    param.topics[d] = (short) ttt;
    tree.update(ttt, (wa[ttt] + beta) / (nk[ttt] + vbeta));
  }

  public void sampleOneWord(TokensAndParams param, FTree tree) {

    int[] wa = param.wtrow;
    buildFTree(param.wtrow, tree);

    Random rand = new Random(System.currentTimeMillis());

    for (int d = 0; d < param.tokens.length; d ++) {
      int did = param.tokens[d];
      sample(param, tree, wa, d, rand, ndk[did]);
    }
  }

//  public void sampleIncreMH(TokensAndParams param, FTree tree, int[] tidx) {
//    int did, tt, ttt;
//    int kk, vv;
//    float psum = 0.0F, u, value;
//
//    int[] wa = param.wtrow;
//    buildFTree(param.wtrow, tree);
//
//    Random rand = new Random(System.currentTimeMillis());
//    S2STightTraverseMap dk;
//
//    boolean incre;
//
//    for (int d = 0; d < param.tokens.length; d ++) {
//      did = param.tokens[d];
//      tt  = param.topics[d];
//      dk  = ndk[did];
//
//      wa[tt] --;
//      nk[tt] --;
//      vv = dk.dec(tt);
//      value = (wa[tt] + beta) / (nk[tt] + vbeta);
//
//      incre = false;
//
//      if (d > 0 && did == param.tokens[d - 1]) {
//        psum -= (vv + 1) * tree.get(tt);
//        psum += vv * value;
//        incre = true;
//      }
//
//      tree.update(tt, value);
//
//
//      if (!incre) {
//        psum = 0.0F;
//        for (int i = 0; i < dk.size; i++) {
//          kk = dk.key[dk.idx[i]];
//          vv = dk.value[dk.idx[i]];
//          psum += vv * tree.get(kk);
//          p[i] = psum;
//          tidx[kk] = i;
//        }
//
//        u = rand.nextFloat() * (psum + alpha * tree.first());
//        if (u < psum) {
//          u = rand.nextFloat() * psum;
//          ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//        } else {
//          ttt = tree.sample(rand.nextFloat() * tree.first());
//        }
//      } else {
//        u = rand.nextFloat() * (psum + alpha * tree.first());
//        if (u < psum) {
//          u = rand.nextFloat() * p[dk.size - 1];
//          ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//
//          // tt -> ttt
//
//          // p(j) = dk.get(ttt) * tree.get(ttt)
//          // q(i) = p[tt] - p[tt - 1]
//          // p(i) = dk.get(tt) * tree.get(tt)
//          // q(j) = p[ttt] - p[ttt - 1]
//
//          float p_j = dk.get(ttt) * tree.get(ttt);
//          float q_i = p[tidx[tt]] - p[tidx[tt] - 1];
//          float p_i = dk.get(tt) * tree.get(tt);
//          float q_j = p[tidx[ttt]] - p[tidx[ttt] - 1];
//
//          if (rand.nextFloat() < ((p_j * q_i) / (p_i * q_j))) {
//
//          } else {
//            ttt = tt;
//          }
//
//
//        } else {
//          ttt = tree.sample(rand.nextFloat() * tree.first());
//        }
//      }
//
//      wa[ttt] ++;
//      nk[ttt] ++;
//      vv = dk.inc(ttt);
//      param.topics[d] = (short) ttt;
//
//      value = (wa[ttt] + beta) / (nk[ttt] + vbeta);
//      if (d < (param.tokens.length - 1) && did == param.tokens[d + 1]) {
//        psum -= (vv - 1) * tree.get(ttt);
//        psum += vv * value;
//      }
//
//      tree.update(ttt, value);
//    }
//  }
//
//  public void sampleIncreStale(TokensAndParams param, FTree tree) {
//    int did, tt, ttt;
//    int kk, vv;
//    float psum = 0.0F, u, value;
//
//    int[] wa = param.wtrow;
//    buildFTree(param.wtrow, tree);
//
//    Random rand = new Random(System.currentTimeMillis());
//    S2STightTraverseMap dk;
//
//    boolean incre;
//
//    for (int d = 0; d < param.tokens.length; d ++) {
//      did = param.tokens[d];
//      tt  = param.topics[d];
//      dk  = ndk[did];
//
//      wa[tt] --;
//      nk[tt] --;
//      vv = dk.dec(tt);
//      value = (wa[tt] + beta) / (nk[tt] + vbeta);
//
//      incre = false;
//
//      if (d > 0 && did == param.tokens[d - 1]) {
//        psum -= (vv + 1) * tree.get(tt);
//        psum += vv * value;
//        incre = true;
//      }
//
//      tree.update(tt, value);
//
//
//      if (!incre) {
//        psum = 0.0F;
//        for (int i = 0; i < dk.size; i++) {
//          kk = dk.key[dk.idx[i]];
//          vv = dk.value[dk.idx[i]];
//          psum += vv * tree.get(kk);
//          p[i] = psum;
//        }
//
//        u = rand.nextFloat() * (psum + alpha * tree.first());
//        if (u < psum) {
//          u = rand.nextFloat() * psum;
//          ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//        } else {
//          ttt = tree.sample(rand.nextFloat() * tree.first());
//        }
//      } else {
//        u = rand.nextFloat() * (psum + alpha * tree.first());
//        if (u < psum) {
//          u = rand.nextFloat() * p[dk.size - 1];
//          ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//        } else {
//          ttt = tree.sample(rand.nextFloat() * tree.first());
//        }
//      }
//
//      wa[ttt] ++;
//      nk[ttt] ++;
//      vv = dk.inc(ttt);
//      param.topics[d] = (short) ttt;
//
//      value = (wa[ttt] + beta) / (nk[ttt] + vbeta);
//      if (d < (param.tokens.length - 1) && did == param.tokens[d + 1]) {
//        psum -= (vv - 1) * tree.get(ttt);
//        psum += vv * value;
//      }
//
//      tree.update(ttt, value);
//    }
//  }
//
//  public void sampleIncremental(TokensAndParams param, FTree tree, int[] tidx, float[] values) {
//    int did, tt, ttt;
//    int kk, vv;
//    float psum, u, value;
//
//    int[] wa = param.wtrow;
//    buildFTree(param.wtrow, tree);
//
//    Random rand = new Random(System.currentTimeMillis());
//    S2STightTraverseMap dk;
//    psum = 0.0F;
//
//    boolean incremental;
//
//
//    for (int d = 0; d < param.tokens.length; d ++) {
//      did = param.tokens[d];
//      tt  = param.topics[d];
//      dk  = ndk[did];
//
//      wa[tt] --;
//      nk[tt] --;
//      vv = dk.dec(tt);
//      value = (wa[tt] + beta) / (nk[tt] + vbeta);
//
//      incremental = false;
//
//      if (d > 0 && did == param.tokens[d - 1]) {
//        psum -= (vv + 1) * tree.get(tt);
//        psum += vv * value;
//        values[tidx[tt]] = vv * value;
//        incremental = true;
//      }
//
//      tree.update(tt, value);
//
//      if (!incremental) {
//        psum = 0.0F;
//        for (int i = 0; i < dk.size; i++) {
//          kk = dk.key[dk.idx[i]];
//          vv = dk.value[dk.idx[i]];
//          values[i] = vv * tree.get(kk);
//          psum += values[i];
//          p[i] = psum;
//          tidx[kk] = i;
//        }
//      }
//
//      u = rand.nextFloat() * (psum + alpha * tree.first());
//
//      if (u < psum) {
//        if (incremental) {
//          p[0] = values[0];
//          for (int i = 1; i < dk.size; i ++) {
//            p[i] = p[i - 1] + values[i];
//          }
//          psum = p[dk.size - 1];
//        }
//
//        u = rand.nextFloat() * psum;
//        ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//      } else {
//        ttt = tree.sample(rand.nextFloat() * tree.first());
//      }
//
//      wa[ttt] ++;
//      nk[ttt] ++;
//      vv = dk.inc(ttt);
//      param.topics[d] = (short) ttt;
//      value = (wa[ttt] + beta) / (nk[ttt] + vbeta);
//
//      if (d < (param.tokens.length - 1) && did == param.tokens[d + 1]) {
//        psum -= (vv - 1) * tree.get(ttt);
//        psum += vv * value;
//        values[tidx[ttt]] = vv * value;
//      }
//
//      tree.update(ttt, value);
//    }
//  }
//
//  public void sampleDelayUpdate(TokensAndParams param, FTree tree) {
//    int did, tt, ttt;
//    int kk, vv;
//    float psum, u;
//
//    int[] wa = param.wtrow;
//    buildFTree(param.wtrow, tree);
//
//    Random rand = new Random(System.currentTimeMillis());
//
//    Random rand2 = new Random(System.currentTimeMillis());
//    S2STightTraverseMap dk;
//
//    for (int d = 0; d < param.tokens.length; d ++) {
//      did = param.tokens[d];
//      tt  = param.topics[d];
//      dk  = ndk[did];
//
//      psum = 0.0F;
//      for (int i = 0; i < dk.size; i ++) {
//        kk = dk.key[dk.idx[i]];
//        vv = dk.value[dk.idx[i]];
//        psum += vv * tree.get(kk);
//        p[i] = psum;
//      }
//
//      u = rand.nextFloat() * (psum + alpha * tree.first());
//      if (u < psum) {
//        u = rand.nextFloat() * psum;
//        ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//      } else {
//        ttt = tree.sample(rand2.nextFloat() * tree.first());
//      }
//
//      if (tt != ttt) {
//        wa[tt] --;
//        nk[tt] --;
//        dk.dec(tt);
//        tree.update(tt, (wa[tt] + beta) / (nk[tt] + vbeta));
//
//        wa[ttt]++;
//        nk[ttt]++;
//        dk.inc(ttt);
//        param.topics[d] = (short) ttt;
//        tree.update(ttt, (wa[ttt] + beta) / (nk[ttt] + vbeta));
//      }
//    }
//  }
//
//  public void sampleDelayIncre(TokensAndParams param, FTree tree) {
//    int did, tt, ttt;
//    int kk, vv;
//    float psum = 0.0F, u, value;
//
//    int[] wa = param.wtrow;
//    buildFTree(param.wtrow, tree);
//
//    Random rand = new Random(System.currentTimeMillis());
//    S2STightTraverseMap dk;
//
//    boolean incre;
//
//    for (int d = 0; d < param.tokens.length; d ++) {
//      did = param.tokens[d];
//      tt  = param.topics[d];
//      dk  = ndk[did];
//
//      incre = false;
//
//      if (d > 0 && did == param.tokens[d - 1])
//        incre = true;
//
//      if (!incre) {
//        psum = 0.0F;
//        for (int i = 0; i < dk.size; i++) {
//          kk = dk.key[dk.idx[i]];
//          vv = dk.value[dk.idx[i]];
//          psum += vv * tree.get(kk);
//          p[i] = psum;
//        }
//
//        u = rand.nextFloat() * (psum + alpha * tree.first());
//        if (u < psum) {
//          u = rand.nextFloat() * psum;
//          ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//        } else {
//          ttt = tree.sample(rand.nextFloat() * tree.first());
//        }
//      } else {
//        u = rand.nextFloat() * (psum + alpha * tree.first());
//        if (u < psum) {
//          u = rand.nextFloat() * p[dk.size - 1];
//          ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
//        } else {
//          ttt = tree.sample(rand.nextFloat() * tree.first());
//        }
//      }
//
//
//      if (tt != ttt) {
//        wa[tt]--;
//        nk[tt]--;
//        vv = dk.dec(tt);
//        value = (wa[tt] + beta) / (nk[tt] + vbeta);
//
//        if (d < (param.tokens.length - 1) && did == param.tokens[d + 1]) {
//          psum -= (vv + 1) * tree.get(tt);
//          psum += vv * value;
//        }
//
//        tree.update(tt, value);
//
//        wa[ttt]++;
//        nk[ttt]++;
//        vv = dk.inc(ttt);
//        param.topics[d] = (short) ttt;
//
//        value = (wa[ttt] + beta) / (nk[ttt] + vbeta);
//        if (d < (param.tokens.length - 1) && did == param.tokens[d + 1]) {
//          psum -= (vv - 1) * tree.get(ttt);
//          psum += vv * value;
//        }
//
//        tree.update(ttt, value);
//      }
//    }
//  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int D = docs.size();
    int V = 12420;
    int K = 1024;
    float alpha = 50.0F / K;
    float beta  = 0.01F;

    FTreeLDA lda = new FTreeLDA(D, V, K, alpha, beta, docs);
    lda.init();

    for (int i = 0; i < 200; i ++) {
      lda.trainOneIteration(i);
    }
  }

  public static void nytimes(String[] argv) throws IOException {
    String path = "data/nytimes.angel";
    List<Document> docs = Utils.read(path);
    int D = docs.size();
    int V = 102661;
    int K = 256;
    float alpha = 50.0F / K;
    float beta  = 0.01F;

    FTreeLDA lda = new FTreeLDA(D, V, K, alpha, beta, docs);
    lda.init();

    for (int i = 0; i < 200; i ++) {
      lda.trainOneIteration(i);
    }
  }

  public static void enron(String[] argv) throws IOException {
    String path = "docword.enron.txt";

    int V = Utils.readV(path);
    int K = 1024;
    List<Document> docs = Utils.read2(path);
    int D = docs.size();

    System.out.format("D=%d V=%d\n", D, V);

    float alpha = 50.0F / K;
    float beta  = 0.01F;

    FTreeLDA lda = new FTreeLDA(D, V, K, alpha, beta, docs);
    lda.init();

    for (int i = 0; i < 100; i ++) {
      lda.trainOneIteration(i);
    }
  }

  public static void main(String[] argv) throws IOException {
//    PropertyConfigurator.configure("conf/log4j.properties");
    nips(argv);
//    enron(argv);
//    nytimes(argv);
  }
}
