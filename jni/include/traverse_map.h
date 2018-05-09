//
//  traverse_map.h
//  cpp_test
//
//  Created by yulele on 16/9/9.
//  Copyright (c) 2016年 yulele. All rights reserved.
//

#ifndef __cpp_test__traverse_map__
#define __cpp_test__traverse_map__

#include <stdio.h>
#include <iostream>
#include "utils.h"

namespace lda {

class traverse_map {
public:
    int * key;
    int * value;
    bool * used;
    int * idx;
    int * poss;
    int mask;
    int n;
    int size;
    
    traverse_map() {
        
    }
    
    void init(int expected) {
        n = next_power_of_two(expected);
        mask = n - 1;
        key = new int[n];
        value = new int[n];
        used = new bool[n];
        idx = new int[n];
        poss = new int[n];
        size = 0;
    }
    
    ~traverse_map() {
        delete [] key;
        delete [] value;
        delete [] used;
        delete [] idx;
        delete [] poss;
    }
    
    int get(int k) {
        // The starting point
        int pos = ( murmur_hash3(k)) & mask;
        
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
    
    inline void put(int k, int v) {
        if (v == 0)
            return;
        
        // The starting point
        int pos = ( murmur_hash3(k)) & mask;
        
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
        idx[size] = pos;
        poss[pos] = size;
        size ++;
    }
    
    void rehash() {
        
        int * kkey = key;
        int * vvalue = value;
        
        key = new int[n];
        value = new int[n];
        
        memset(used, 0, n * sizeof(bool));
        
        int temp = size;
        size = 0;
        
        for (int i = 0; i < temp; i ++) {
            int k = kkey[idx[i]];
            int v = vvalue[idx[i]];
            put(k, v);
        }
        
        delete [] kkey;
        delete [] vvalue;
        
    }
    
    void dec(int k) {
        int pos = ( murmur_hash3(k)) & mask;
        
        while (used[pos]) {
            if (key[pos] == k) {
                value[pos] --;
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
    
    void inc(int k) {
        int pos = ( murmur_hash3(k)) & mask;
        
        int cnt = 0;
        while (used[pos]) {
            if (key[pos] == k) {
                value[pos] ++;
                if (value[pos] == 1) {
                    idx[size] = pos;
                    poss[pos] = size;
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
        used[pos] = true;
        idx[size] = pos;
        poss[pos] = size;
        size ++;
    }
};
    
}

#endif /* defined(__cpp_test__traverse_map__) */
