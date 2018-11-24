#include <stdio.h>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "params/fc1_b.txt"
#include "params/fc1_w.txt"
#include "params/fc2_b.txt"
#include "params/fc2_w.txt"
#include "params/fc3_b.txt"
#include "params/fc3_w.txt"
#include "params/I_b.txt"
#include "params/I_w.txt"
#include "params/rnn1_bi.txt"
#include "params/rnn1_bh.txt"
#include "params/rnn1_wi.txt"
#include "params/rnn1_wh.txt"
#include "params/rnn2_bi.txt"
#include "params/rnn2_bh.txt"
#include "params/rnn2_wi.txt"
#include "params/rnn2_wh.txt"

#define SAMPLE_SIZE 92675
#define HIDDEN_SIZE 256
#define MELS_DIM 80
#define AUX_DIM 32

using namespace std;
using TYPE = double;

template<int n> class Array {
private:
    TYPE arr[n] = {};
    TYPE max = 0;
    TYPE min = 0;
public:
    TYPE& operator[](int i) {
        if (max < arr[i]) max = arr[i];
        if (min > arr[i]) min = arr[i];
        return arr[i];
    }

    const TYPE& operator[](int i) const {
        return arr[i];
    }

    const int getSize() const {
        return n;
    }

    void print(string s) {
        printf("[%s] max:%.16f, min=%.16f\n", s.c_str(), max, min);
    }
};

Array<SAMPLE_SIZE> out;
Array<1 + MELS_DIM + AUX_DIM> I;
Array<HIDDEN_SIZE + AUX_DIM> inp;
Array<HIDDEN_SIZE> x;
Array<HIDDEN_SIZE> p;
Array<HIDDEN_SIZE> h1;
Array<HIDDEN_SIZE> h2;
TYPE sample = 0;

Array<HIDDEN_SIZE * 3> igates;
Array<HIDDEN_SIZE * 3> hgates;
Array<HIDDEN_SIZE> reset_gate;
Array<HIDDEN_SIZE> input_gate;
Array<HIDDEN_SIZE> new_gate;

TYPE mels[SAMPLE_SIZE][MELS_DIM];
TYPE aux_0[SAMPLE_SIZE][AUX_DIM];
TYPE aux_1[SAMPLE_SIZE][AUX_DIM];
TYPE aux_2[SAMPLE_SIZE][AUX_DIM];
TYPE aux_3[SAMPLE_SIZE][AUX_DIM];

void debug() {
    out.print("out");
    I.print("I");
    inp.print("inp");
    x.print("x");
    p.print("p");
    h1.print("h1");
    h2.print("h2");
    igates.print("igates");
    hgates.print("hgates");
    reset_gate.print("reset_gate");
    input_gate.print("input_gate");
    new_gate.print("new_gate");
}

void savefile() {
    FILE *fp;
    fp = fopen("output.txt", "w");
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        fprintf(fp, "%.16f\n", out[i]);
    }
    fclose(fp);
}

void loadfile() {
    // mels
    thread t1([]() {
        FILE *fp_mels;
        fp_mels = fopen("mels.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < MELS_DIM; j++) {
                fscanf(fp_mels, "%lf", &mels[i][j]);
            }
        }
        fclose(fp_mels);
    });

    // aux_0
    thread t2([]() {
        FILE *fp_aux_0;
        fp_aux_0 = fopen("aux_0.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_0, "%lf", &aux_0[i][j]);
            }
        }
        fclose(fp_aux_0);
    });

    // aux_1
    thread t3([]() {
        FILE *fp_aux_1;
        fp_aux_1 = fopen("aux_1.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_1, "%lf", &aux_1[i][j]);
            }
        }
        fclose(fp_aux_1);
    });

    // aux_2
    thread t4([]() {
        FILE *fp_aux_2;
        fp_aux_2 = fopen("aux_2.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_2, "%lf", &aux_2[i][j]);
            }
        }
        fclose(fp_aux_2);
    });

    // aux_3
    thread t5([]() {
        FILE *fp_aux_3;
        fp_aux_3 = fopen("aux_3.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_3, "%lf", &aux_3[i][j]);
            }
        }
        fclose(fp_aux_3);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
}

template<class T> inline T exp_d(const T x) {
    T tmp = 1. + x / 1024.;
    for (int i = 0; i < 10; i++) {
        tmp *= tmp;
    }
    return tmp;
}

template<class T> inline T sigmoid(const T x) {
    return 1. / (1. + exp_d(-x));
}

template<class T> inline T tanh(const T x) {
    T plus = exp_d(x);
    T nega = exp_d(-x);
    return (plus - nega) / (plus + nega);
}

template<class T> void add(T &out, const T &in) {
    for (int i = 0; i < out.getSize(); i++) {
        out[i] += in[i];
    }
}

template<class T> void relu(T &x) {
    for (int i = 0; i < x.getSize(); i++) {
        x[i] = (x[i] < 0) ? 0 : x[i];
    }
}

template<class T> void softmax(T &x) {
    TYPE sum = 0;
    TYPE tmp;
    for (int i = 0; i < x.getSize(); i++) {
        tmp = exp_d(x[i]);
        x[i] = tmp;
        sum += tmp;
    }

    for (int i = 0; i < x.getSize(); i++) {
        x[i] /= sum;
    }
}

template<class T> int choice(const T &x) {
    random_device rnd;
    mt19937 mt(rnd());
    uniform_real_distribution<TYPE> dist(0.0, 1.0);
    TYPE threshold = dist(mt);

    for (int i = 0; i < x.getSize(); i++) {
        if (threshold < x[i]) return i;
        threshold -= x[i];
    }
    return 0;
}

template<class T1, class T2, int p1, int p2> void concat(T1 &out, const T2 v, const T2 (&in1)[p1], const T2 (&in2)[p2]) {
    out[0] = v;

    for (int i = 0; i < p1; i++) {
        out[1 + i] = in1[i];
    }

    for (int i = 0; i < p2; i++) {
        out[1 + p1 + i] = in2[i];
    }
}

template<class T1, class T2, class T3, int p2> void concat_2(T1 &out, const T2 &in1, const T3 (&in2)[p2]) {
    int p1 = in1.getSize();

    for (int i = 0; i < p1; i++) {
        out[i] = in1[i];
    }

    for (int i = 0; i < p2; i++) {
        out[p1 + i] = in2[i];
    }
}

template<class T1, class T2, class T3, int r, int c> void linear(T1 &out, const T2 &in, const T3 (&w)[r][c], const T3 (&b)[r]) {
    TYPE sum;
    for (int i = 0; i < r; i++) {
        sum = 0;
        for (int j = 0; j < c; j++) {
            sum += in[j] * w[i][j];
        }
        out[i] = sum + b[i];
    }
}

template<class T1, class T2, class T3, int r, int ci, int ch> void gru(const T1 &x, T2 &h, const T3 (&wi)[r][ci], const T3 (&wh)[r][ch], const T3 (&bi)[r], const T3 (&bh)[r]) {
    // igates, hgates
    linear(igates, x, wi, bi);
    linear(hgates, h, wh, bh);

    // reset_gate, input_gate, new_gate
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        reset_gate[i] = sigmoid(igates[i] + hgates[i]);
        input_gate[i] = sigmoid(igates[HIDDEN_SIZE + i] + hgates[HIDDEN_SIZE + i]);
        new_gate[i] = tanh(igates[HIDDEN_SIZE * 2 + i] + (reset_gate[i] * hgates[HIDDEN_SIZE * 2 + i]));

        // h_next
        h[i] = new_gate[i] + (input_gate[i] * (h[i] - new_gate[i]));
    }
}

int main() {
    printf("***** Start WaveRNN inference *****\n");

    // load inputs
    printf("Loading inputs from file...\n");
    loadfile();

    // inference loop
    auto start = chrono::system_clock::now();
    printf("Enter inference loop!!!\n");
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        // I
        concat(I, sample, mels[i], aux_0[i]);
        linear(x, I, I_w, I_b);

        // rnn1
        gru(x, h1, rnn1_wi, rnn1_wh, rnn1_bi, rnn1_bh);
        add(x, h1);

        // rnn2
        concat_2(inp, x, aux_1[i]);
        gru(inp, h2, rnn2_wi, rnn2_wh, rnn2_bi, rnn2_bh);
        add(x, h2);

        // fc1
        concat_2(inp, x, aux_2[i]);
        linear(x, inp, fc1_w, fc1_b);
        relu(x);

        // fc2
        concat_2(inp, x, aux_3[i]);
        linear(x, inp, fc2_w, fc2_b);
        relu(x);

        // fc3
        linear(p, x, fc3_w, fc3_b);
        softmax(p);

        // categorize
        sample = 2 * choice(p) / (HIDDEN_SIZE - 1.) - 1.;
        out[i] = sample;

        // show progress
        if (((i + 1) % (SAMPLE_SIZE / 10)) == 0) {
            auto end = chrono::system_clock::now();
            auto sec = chrono::duration<TYPE>(end - start).count();
            printf("|%7.2lf s||%3d %%|", sec, ((i + 1) / (SAMPLE_SIZE / 100)));
            for (int j = 0; j < ((i + 1) / (SAMPLE_SIZE / 10)); j++) {
                printf("##");
            }
            printf("\n");
        }
    }

    // save outputs
    printf("Saving outputs to file...\n");
    savefile();

    // debug
    debug();

    printf("***** Finish WaveRNN inference *****\n");

    return 0;
}
