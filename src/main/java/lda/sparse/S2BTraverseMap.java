package lda.sparse;

import it.unimi.dsi.fastutil.HashCommon;

import java.util.Arrays;

/**
 * Created by leleyu on 2016/10/29.
 */
public class S2BTraverseMap extends TraverseHashMap {

  public byte[] value;
  boolean[] used;
  public byte[] idx;
  byte[] poss;

  public S2BTraverseMap(byte expected) {
    super(expected);
    value = new byte[n];
    used  = new boolean[n];
    idx   = new byte[n];
    poss  = new byte[n];
  }

  public S2BTraverseMap(int expected) {
    this((byte) expected);
  }

  @Override
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

  @Override
  public short get(int k) {
    return get((short) k);
  }

  @Override
  public void put(short k, short v) {
    put(k, (byte) v);
  }

  public void put(short k, byte v) {
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
    idx[size] = (byte) pos;
    poss[(byte) pos] = (byte) size;
    size ++;
  }

  @Override
  public void rehash() {
    short[] kkey = key;
    byte [] vvalue = value;

    key = new short[n];
    value = new byte[n];

    Arrays.fill(used, false);

    int temp = size;
    size = 0;

    for (int i = 0; i < temp; i ++) {
      short k = kkey[idx[i]];
      byte v = vvalue[idx[i]];
      put(k, v);
    }

  }

  @Override
  public short dec(short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    while (used[pos]) {
      if (key[pos] == k) {
        value[pos] --;
        if (value[pos] == 0) {
          size --;
          idx[poss[pos]] = idx[size];
          poss[idx[size]] = poss[pos];
        }
        return value[pos];
      }

      pos = (pos + 1) & mask;
    }
    return 0;
  }

  @Override
  public short dec(int k) {
    return dec((short) k);
  }

  @Override
  public short inc(short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    int cnt = 0;
    while (used[pos]) {
      if (key[pos] == k) {
        value[pos] ++;
        if (value[pos] == 1) {
          idx[size] = (byte) pos;
          poss[(byte) pos] = (byte) size;
          size ++;
        }

        return value[pos];
      }

      cnt ++;
      if (cnt > n) {
        rehash();
        return inc(k);
      }
      pos = (pos + 1) & mask;
    }

    key[pos] = k;
    value[pos] = 1;
    used[pos] = true;
    idx[size] = (byte) pos;
    poss[(byte) pos] = (byte) size;
    size ++;
    return 1;
  }

  @Override
  public short inc(int k) {
    return inc((short) k);
  }

  @Override
  public int size() {
    int sum = 0;
    sum += key.length * 2;
    sum += value.length;
    sum += used.length;
    sum += idx.length;
    sum += poss.length;
    return sum;
  }

  @Override
  public short getKey(int idx) {
    return key[this.idx[idx]];
  }

  @Override
  public short getVal(int idx) {
    return value[this.idx[idx]];
  }
}
