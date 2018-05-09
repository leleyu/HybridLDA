package lda.utils;

/**
 * Created by leleyu on 2016/1/28.
 * Binary index tree only for trees whose size is a mutilpe of 2.
 */
public class FTree {

  public float[] tree;

  int length;

  int K;

  public FTree(int length) {
    int len = nextPowerOfTwo(length);
    tree = new float[2 * len];
    this.length = len;
    this.K = length;
  }

  public FTree(float[] p, int length) {
    this(length);
    build(p);
  }

  public void build(float[] p) {
    int start = Math.min(2 * length - 1, length + p.length - 1);
    for (int i = start; i > 0; i --)  {
      if (i >= length) {
        tree[i] = p[i - length];
      }
      else {
        tree[i] = tree[i << 1] + tree[(i << 1) + 1];
      }
    }
  }

  public void update(int index, float value) {
    int i = index + length;
    float delta = value - tree[i];
    while (i > 0) {
      tree[i] += delta;
      i >>= 1;
    }
  }

  public int sample(float u) {
    int i = 1;
    while (i < length) {
      if (u < tree[i << 1]) {
        i <<= 1;
      } else {
        u = u - tree[i << 1];
        i = i * 2 + 1;
      }
    }
    return Math.min(i - length, K - 1);
  }

  public static int nextPowerOfTwo(int x) {
    if(x == 0) {
      return 1;
    } else {
      --x;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      return (x | x >> 16) + 1;
    }
  }

  public float first() {
    return tree[1];
  }

  public float get(int index) {
    return tree[index + length];
  }

  public static void main(String[] argv) {
    System.out.println(nextPowerOfTwo(1024));
  }

}
