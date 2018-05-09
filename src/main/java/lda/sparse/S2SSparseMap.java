package lda.sparse;

import it.unimi.dsi.fastutil.HashCommon;

import java.util.Arrays;

/**
 * Created by leleyu on 2016/11/5.
 */
public class S2SSparseMap {
  public short size;
  public short[] key;
  public short[] value;
  public short n;
  public int mask;

  public S2SSparseMap(int expected) {
    this.n = (short) HashCommon.arraySize(expected, 1.0F);
    this.key = new short[n];
    this.value = new short[n];
    this.size = 0;
    this.mask = n - 1;
    Arrays.fill(key, (short) -1);
  }

  public boolean used(int pos) {
    return key[pos] != -1;
  }

  public void put(final short k, final short v) {
    if (v == 0)
      return;

    // The starting point
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    // There's always an unused entry.
    while (used(pos)) {
      if (key[pos] == k) {
        value[pos] = v;
        return;
      }
      pos = (pos + 1) & mask;
    }

    key[pos] = k;
    value[pos] = v;
    size ++;
  }

  public void rehash() {

    short[] kkey = key;
    short[] vvalue = value;

    key = new short[n];
    value = new short[n];

    Arrays.fill(key, (short) -1);
    size = 0;

    for (int i = 0; i < n; i ++) {
      if (kkey[i] != -1) {
        short k = kkey[i];
        short v = vvalue[i];
        put(k, v);
      }
    }

  }

  public short dec(final short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    while (used(pos)) {
      if (key[pos] == k) {
        value[pos] --;
        if (value[pos] == 0)
          size --;
        return value[pos];
      }

      pos = (pos + 1) & mask;
    }
    return 0;
  }

  public short dec(int k) {
    return dec((short) k);
  }

  public short inc(final short k) {
    int pos = ( HashCommon.murmurHash3(k)) & mask;

    int cnt = 0;
    while (used(pos)) {
      if (key[pos] == k) {
        value[pos] ++;
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
    size ++;
    return 1;
  }

  public short inc(int k) {
    return inc((short) k);
  }
}
