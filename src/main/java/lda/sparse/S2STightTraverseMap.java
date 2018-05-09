package lda.sparse;

import it.unimi.dsi.fastutil.HashCommon;

import java.util.Arrays;

/**
 * Created by leleyu on 2016/7/29.
 */
public class S2STightTraverseMap extends TraverseHashMap {

//  public short[] value;
//  boolean[] used;
//  public short[] idx;
//  short[] poss;


  short[] values;

//  int mask;
//  public short n;

//  public float norm = 0.0F;

  public S2STightTraverseMap(short expected) {
//    this.n = (short) HashCommon.arraySize(expected, 1.0F);
//    mask =  n - 1;
//    key   = new short[n];
//    value = new short[n];
//    used  = new boolean[n];
//    idx   = new short[n];
//    poss  = new short[n];

    super(expected);
    values = new short[n * 3];
  }

  public S2STightTraverseMap(int expected) {
    this((short) expected);
  }

  public short get(final int k) {
    return get((short) k);
  }

  public short get(final short k) {
    // The starting point
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    // There's always an unused entry.
    int cnt = 0;
    while (used(pos)) {
      if (key[pos] == k) {
        return values[pos];
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
    while (used(pos)) {
      if (key[pos] == k) {
        values[pos] = v;
        return;
      }
      pos = (pos + 1) & mask;
    }

//    used[pos] = true;
    key[pos] = k;
    values[pos] = v;
    values[size + n] = (short) pos;
    values[(short) pos + n * 2] = size;
    size ++;
  }

  public void rehash() {

    short[] kkey = key;
    short[] vvalue = values;

//    print();

    key = new short[n];
    values = new short[n * 3];

    Arrays.fill(key, (short) -1);

    int temp = size;
    size = 0;

    for (int i = 0; i < temp; i ++) {
      short k = kkey[vvalue[idx(i)]];
      short v = vvalue[vvalue[idx(i)]];
      put(k, v);

//      if (key[idx[i]] != k) {
//        System.out.format("Error key[idx[%d]]=%d while expect %d\n", i, idx[i], k);
//      }
    }


//    for (int i = 0; i < kkey.length; i ++) {
//      if (vvalue[i] != 0) {
//        put(kkey[i], vvalue[i]);
//      }
//    }
//    System.out.format("Rehash before size %d after size %d n %d\n", temp, size, n);

  }

  public short dec(final int k) {
    return dec((short) k);
  }

  public short dec(final short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    while (used(pos)) {
      if (key[pos] == k) {
//        norm -= value[pos] * value[pos];
        values[pos] --;
//        norm += value[pos] * value[pos];
        if (values[pos] == 0) {
          size --;
          values[idx(values[poss(pos)])] = values[idx(size)];
          values[poss(values[idx(size)])] = values[poss(pos)];
        }
        return values[pos];
      }

      pos = (pos + 1) & mask;
    }
    return 0;
  }

  public short inc(final int k) {
    return inc((short) k);
  }

  public short inc(final short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    int cnt = 0;
    while (used(pos)) {
      if (key[pos] == k) {
//        norm -= value[pos] * value[pos];
        values[pos] ++;
//        norm += value[pos] * value[pos];
        if (values[pos] == 1) {
          values[idx(size)] = (short) pos;
          values[poss(pos)] = size;
          size ++;
        }

        return values[pos];
      }

      cnt ++;
      if (cnt > n) {
        rehash();
        return inc(k);
      }
      pos = (pos + 1) & mask;
    }

    key[pos] = k;
    values[pos] = 1;
//    norm += 1;
    values[idx(size)] = (short) pos;
    values[poss(pos)] = size;
    size ++;
    return 1;
  }

//  public void checkNorm() {
//    double sum = 0.0;
//    for (int i = 0; i < size; i ++) {
//      sum += value[idx[i]] * value[idx[i]];
//    }
//
//    if (Math.abs(sum - norm) > 10e-9) {
//      System.out.format("check norm error sum=%f while norm=%f\n", sum, norm);
//    }
//  }

  @Override
  public int size() {
    int sum = 0;
    sum += key.length * 2;
    sum += values.length * 2;
//    sum += used.length;
    return sum;
  }

  @Override
  public short getKey(int idx) {
    return key[values[idx(idx)]];
  }

  @Override
  public short getVal(int idx) {
    return values[values[idx(idx)]];
  }


  public static void main(String[] argv) {
    for (int i = 0; i < 1024; i ++)
      System.out.println(i + " " + HashCommon.murmurHash3(i));
  }
}
