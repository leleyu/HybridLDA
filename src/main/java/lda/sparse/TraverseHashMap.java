package lda.sparse;

import it.unimi.dsi.fastutil.HashCommon;

import java.util.Arrays;

/**
 * Created by leleyu on 2016/10/25.
 */
public abstract class TraverseHashMap {

  public short size;
  public short[] key;
  public short n;
  public int mask;

  public TraverseHashMap(int expected) {
    this.n = (short) HashCommon.arraySize(expected, 1.0F);
    if (n <= 0) {
      System.out.println("expected=" + expected + " n=" + n);
    }
    this.key = new short[n];
    this.size = 0;
    this.mask = n - 1;
    Arrays.fill(key, (short) -1);
  }

  public TraverseHashMap() {

  }

  public abstract short get(final short k);

  public abstract short get(final int k);

  public abstract void put(final short k, final short v);

  public abstract void rehash();

  public abstract short dec(final short k);

  public abstract short dec(final int k);

  public abstract short inc(final short k);

  public abstract short inc(final int k);

  public abstract int size();

  public abstract short getKey(int idx);

  public abstract short getVal(int idx);

  protected int idx(int pos) {
    return n + pos;
  }

  protected int poss(int pos) {
    return n * 2 + pos;
  }

  protected boolean used(int pos) {
    return key[pos] != -1;
  }



}
