package lda.utils;

import it.unimi.dsi.fastutil.HashCommon;

import java.util.Arrays;

/**
 * Created by leleyu on 2016/7/29.
 */
public class TraverseMap {

  public short[] key;
  public short[] value;
  boolean[] used;
  public short[] idx;
  short[] poss;
  public short size;
  int mask;
  public short n;

  public double norm = 0.0F;

  public TraverseMap(short expected) {
    this.n = (short) HashCommon.arraySize(expected, 1.0F);
    mask =  n - 1;
    key   = new short[n];
    value = new short[n];
    used  = new boolean[n];
    idx   = new short[n];
    poss  = new short[n];
  }

  public TraverseMap(int expected) {
    this((short) expected);
  }

  public short get(int k) {
    return get((short) k);
  }

  public short get(short k) {
    // The starting point
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    // There's always an unused entry.
    int cnt = 0;
    while (used[pos]) {
      if (key[pos] == k) {
        return value[pos];
      }
      pos = (pos + 1) & mask;
      cnt ++;

      if (cnt > n) {
        rehash();
        return get(k);
      }
    }
    return 0;
  }

  public void put(final int k, final int v) {
    put((short) k, (short) v);
  }

  public void put(final short k, final short v) {
    if (v == 0)
      return;

    // The starting point
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    // There's always an unused entry.
    while (used[pos]) {
      if (key[pos] == k) {
        value[pos] = v;
        return;
      }
      pos = (pos + 1) & mask;
    }

    used[pos] = true;
    key[pos] = k;
    value[pos] = v;
    idx[size] = (short) pos;
    poss[(short) pos] = size;
    size ++;
  }

  public void rehash() {

    short[] kkey = key;
    short[] vvalue = value;

    key = new short[n];
    value = new short[n];

    Arrays.fill(used, false);

    int temp = size;
    size = 0;

    for (int i = 0; i < temp; i ++) {
      short k = kkey[idx[i]];
      short v = vvalue[idx[i]];
      put(k, v);
    }

  }

  public void dec(final int k) {
    dec((short) k);
  }

  public void dec(final short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    while (used[pos]) {
      if (key[pos] == k) {
        norm -= value[pos] * value[pos];
        value[pos] --;
        norm += value[pos] * value[pos];
        if (value[pos] == 0) {
          size --;
          idx[poss[pos]] = idx[size];
          poss[idx[size]] = poss[pos];
        }
        return;
      }

      pos = (pos + 1) & mask;
    }
  }

  public void inc(final int k) {
    inc((short) k);
  }

  public void inc(final short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    int cnt = 0;
    while (used[pos]) {
      if (key[pos] == k) {
        norm -= value[pos] * value[pos];
        value[pos] ++;
        norm += value[pos] * value[pos];
        if (value[pos] == 1) {
          idx[size] = (short) pos;
          poss[(short) pos] = size;
          size ++;
        }

        return;
      }

      cnt ++;
      if (cnt > n) {
        rehash();
        inc(k);
        return;
      }
      pos = (pos + 1) & mask;
    }

    key[pos] = k;
    value[pos] = 1;
    norm += 1;
    used[pos] = true;
    idx[size] = (short) pos;
    poss[(short) pos] = size;
    size ++;
  }

  public void checkNorm() {
    double sum = 0.0;
    for (int i = 0; i < size; i ++) {
      sum += value[idx[i]] * value[idx[i]];
    }

    if (Math.abs(sum - norm) > 10e-9) {
      System.out.format("check norm error sum=%f while norm=%f\n", sum, norm);
    }
  }


  public static void main(String[] argv) {
    for (int i = 0; i < 1024; i ++)
      System.out.println(i + " " + HashCommon.murmurHash3(i));
  }
}
