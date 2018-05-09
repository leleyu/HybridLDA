package lda.sparse;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by leleyu on 2016/12/24.
 */
public class S2STraverseArray extends TraverseHashMap {

  public short[] value;
  public short[] idx;
  public short[] poss;

  public S2STraverseArray(int expected) {
//    super(expected);
    this.n = (short) expected;
    this.value = new short[n];
    this.idx   = new short[n];
    this.poss  = new short[n];
    this.size  = 0;
  }

  @Override
  public short get(short k) {
    return value[k];
  }

  @Override
  public short get(int k) {
    return value[k];
  }

  @Override
  public void put(short k, short v) {
    value[k] = v;
  }

  @Override
  public void rehash() {
    // Do nothing.
  }

  @Override
  public short dec(short k) {
    value[k] --;
    if (value[k] == 0) {
      size --;
      idx[poss[k]] = idx[size];
      poss[idx[size]] = poss[k];
    }
    return value[k];
  }

  @Override
  public short dec(int k) {
    return dec((short) k);
  }

  @Override
  public short inc(short k) {
    value[k] ++;
    if (value[k] == 1) {
      poss[k] = size;
      idx[size ++] = k;
    }
    return value[k];
  }

  @Override
  public short inc(int k) {
    return inc((short) k);
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public short getKey(int idx) {
    return this.idx[idx];
  }

  @Override
  public short getVal(int idx) {
    return value[this.idx[idx]];
  }

  public static void main(String[] argv) {
    int K = 1024;
    int N = 10000;
    short[][] buf = new short[N][];

    for (int i = 0; i < N; i ++)
      buf[i] = new short[K];

    int[] idx = new int[1000];

    Random rand = new Random(System.currentTimeMillis());
    for (int i = 0; i < 1000; i ++) {
      idx[i] = rand.nextInt(1000);
    }




  }
}
