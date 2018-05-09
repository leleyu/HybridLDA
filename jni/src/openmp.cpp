//
//  openmp.cpp
//  cpp_test
//
//  Created by yulele on 16/9/10.
//  Copyright (c) 2016年 yulele. All rights reserved.
//

#include <stdio.h>
#include <omp.h>
#include <algorithm>

int main(int argc, char * argv[]) {
    #pragma omp parallel num_threads(5)
    {
        printf("Hello World!\n");
    }

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 16; i ++)
        printf("thread_num=%d i=%d\n", omp_get_thread_num(), i);

    for (double i = 0; i < 1; i +=0.01)
        printf("%lf %lf\n", i, lgamma(i));
    return 0;
}


