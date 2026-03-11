// fuzzy_sd.hpp
// Реализация Mamdani FIS "Fuzzy_SD" (как в .fis) на C++ без зависимостей.
//
// System:
//  - Type: mamdani
//  - AndMethod=min, OrMethod=max
//  - ImpMethod=min, AggMethod=max
//  - DefuzzMethod=centroid
//
// Inputs:
//  V  in [0,30]    : trapmf same[-11.25 0 2 10], different[5 15 30 31]
//  d  in [0,20]    : trapmf close[1 3 6 8], far[5 10 20 21], trimf s_close[-1 0 4]
//  fi in [0,180]   : trapmf same[-1 0 10 50], perpendicular[40 75 105 135], opposite[115 150 180 190]
//  C  in [0,4]     : trimf animal[2 3 4], bicycle[0 1 2], human[1 2 3], undefinded[3 4 5]
//
// Output:
//  class in [0,1]  : trimf Safe[-0.3952 0 0.9], Danger[0.1 1 2.075]
//
// Rules: 72 (вшиты ниже)

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

class FuzzySD {
public:
    // grid_n: число точек для центроид-дефаззификации на [0..1]
    explicit FuzzySD(std::size_t grid_n = 301) : grid_n_(grid_n < 2 ? 2 : grid_n) {}

    // Возвращает число 0..1
    double eval(double V, double d, double fi, double C) const {
        // 1) Clamp входов к диапазонам из FIS (как минимум чтобы не улетать)
        V  = clamp(V,  0.0, 30.0);
        d  = clamp(d,  0.0, 20.0);
        fi = clamp(fi, 0.0, 180.0);
        C  = clamp(C,  0.0, 4.0);

        // 2) Степени принадлежности входов
        // Input1: V (2 MFs)
        const double muV_same      = trapmf(V,  -11.25, 0.0,  2.0, 10.0);
        const double muV_diff      = trapmf(V,   5.0,  15.0, 30.0, 31.0);
        const std::array<double, 2> muV{muV_same, muV_diff};

        // Input2: d (3 MFs)
        const double muD_close     = trapmf(d,   1.0,  3.0,  6.0,  8.0);
        const double muD_far       = trapmf(d,   5.0, 10.0, 20.0, 21.0);
        const double muD_sclose    = trimf(d,   -1.0,  0.0,  4.0);
        const std::array<double, 3> muD{muD_close, muD_far, muD_sclose};

        // Input3: fi (3 MFs)
        const double muFi_same     = trapmf(fi, -1.0,  0.0, 10.0, 50.0);
        const double muFi_perp     = trapmf(fi, 40.0, 75.0,105.0,135.0);
        const double muFi_opp      = trapmf(fi,115.0,150.0,180.0,190.0);
        const std::array<double, 3> muFi{muFi_same, muFi_perp, muFi_opp};

        // Input4: C (4 MFs)
        const double muC_animal    = trimf(C,   2.0, 3.0, 4.0);
        const double muC_bicycle   = trimf(C,   0.0, 1.0, 2.0);
        const double muC_human     = trimf(C,   1.0, 2.0, 3.0);
        const double muC_undef     = trimf(C,   3.0, 4.0, 5.0);
        const std::array<double, 4> muC{muC_animal, muC_bicycle, muC_human, muC_undef};

        // 3) Mamdani inference:
        //    firing = min( muV[v-1], muD[d-1], muFi[fi-1], muC[c-1] )
        //    implication: min(firing, mu_out(y))
        //    aggregation per-output MF: max over rules -> alpha_safe / alpha_danger
        double alpha_safe   = 0.0; // Output MF index 1
        double alpha_danger = 0.0; // Output MF index 2

        for (const auto& r : rules_) {
            const double firing = std::min(
                std::min(muV[r.iV - 1], muD[r.iD - 1]),
                std::min(muFi[r.iFi - 1], muC[r.iC - 1])
            ) * r.weight; // weight=1 в твоём FIS

            if (r.out == 1) alpha_safe   = std::max(alpha_safe,   firing);
            else            alpha_danger = std::max(alpha_danger, firing);
        }

        // 4) Defuzz (centroid) на диапазоне Output [0..1]
        //    mu_agg(y) = max( min(alpha_safe, muSafe(y)), min(alpha_danger, muDanger(y)) )
        const double y_min = 0.0, y_max = 1.0;
        const double step = (y_max - y_min) / double(grid_n_ - 1);

        double num = 0.0;
        double den = 0.0;

        for (std::size_t i = 0; i < grid_n_; ++i) {
            const double y = y_min + step * double(i);
            const double mu_safe   = trimf(y, -0.3952, 0.0, 0.9);
            const double mu_danger = trimf(y,  0.1,    1.0, 2.075);

            const double imp_safe   = std::min(alpha_safe,   mu_safe);
            const double imp_danger = std::min(alpha_danger, mu_danger);

            const double mu = std::max(imp_safe, imp_danger);

            num += y * mu;
            den += mu;
        }

        // Если ничего не активировалось (den==0), вернём нейтральное значение.
        // (В MATLAB обычно тоже получится что-то около середины при "пустом" выводе.)
        if (den <= 1e-12) return 0.5;

        // Центроид
        return clamp(num / den, 0.0, 1.0);
    }

private:
    struct Rule {
        int iV;   // 1..2
        int iD;   // 1..3
        int iFi;  // 1..3
        int iC;   // 1..4
        int out;  // 1..2 (Safe/Danger)
        double weight;
    };

    std::size_t grid_n_;

    static double clamp(double x, double a, double b) {
        return std::max(a, std::min(b, x));
    }

    // trapmf: [a b c d]
    // 0 outside [a,d], ramps a->b and c->d, plateau [b,c]
    static double trapmf(double x, double a, double b, double c, double d) {
        if (x <= a || x >= d) return 0.0;
        if (x >= b && x <= c) return 1.0;
        if (x > a && x < b)   return (x - a) / (b - a);
        // x in (c, d)
        return (d - x) / (d - c);
    }

    // trimf: [a b c]
    static double trimf(double x, double a, double b, double c) {
        if (x <= a || x >= c) return 0.0;
        if (x == b) return 1.0;
        if (x > a && x < b) return (x - a) / (b - a);
        // x in (b, c)
        return (c - x) / (c - b);
    }

    // 72 rules exactly as in your FIS listing
    static constexpr std::array<Rule, 72> rules_ = {{
        // V=1 (same), d=1 (close), fi=1
        {1,1,1,1, 2,1},{1,1,1,2, 1,1},{1,1,1,3, 1,1},{1,1,1,4, 2,1},
        // V=1, d=1, fi=2
        {1,1,2,1, 2,1},{1,1,2,2, 1,1},{1,1,2,3, 2,1},{1,1,2,4, 2,1},
        // V=1, d=1, fi=3
        {1,1,3,1, 2,1},{1,1,3,2, 2,1},{1,1,3,3, 2,1},{1,1,3,4, 2,1},

        // V=1, d=2 (far), fi=1
        {1,2,1,1, 1,1},{1,2,1,2, 1,1},{1,2,1,3, 1,1},{1,2,1,4, 1,1},
        // V=1, d=2, fi=2
        {1,2,2,1, 1,1},{1,2,2,2, 1,1},{1,2,2,3, 1,1},{1,2,2,4, 1,1},
        // V=1, d=2, fi=3
        {1,2,3,1, 1,1},{1,2,3,2, 1,1},{1,2,3,3, 1,1},{1,2,3,4, 2,1},

        // V=1, d=3 (s_close), fi=1
        {1,3,1,1, 2,1},{1,3,1,2, 2,1},{1,3,1,3, 2,1},{1,3,1,4, 2,1},
        // V=1, d=3, fi=2
        {1,3,2,1, 2,1},{1,3,2,2, 2,1},{1,3,2,3, 2,1},{1,3,2,4, 2,1},
        // V=1, d=3, fi=3
        {1,3,3,1, 2,1},{1,3,3,2, 2,1},{1,3,3,3, 2,1},{1,3,3,4, 2,1},

        // V=2 (different), d=1 (close), fi=1
        {2,1,1,1, 2,1},{2,1,1,2, 1,1},{2,1,1,3, 1,1},{2,1,1,4, 2,1},
        // V=2, d=1, fi=2
        {2,1,2,1, 2,1},{2,1,2,2, 1,1},{2,1,2,3, 1,1},{2,1,2,4, 2,1},
        // V=2, d=1, fi=3
        {2,1,3,1, 2,1},{2,1,3,2, 2,1},{2,1,3,3, 2,1},{2,1,3,4, 2,1},

        // V=2, d=2 (far), fi=1
        {2,2,1,1, 1,1},{2,2,1,2, 1,1},{2,2,1,3, 1,1},{2,2,1,4, 1,1},
        // V=2, d=2, fi=2
        {2,2,2,1, 1,1},{2,2,2,2, 1,1},{2,2,2,3, 1,1},{2,2,2,4, 1,1},
        // V=2, d=2, fi=3
        {2,2,3,1, 2,1},{2,2,3,2, 1,1},{2,2,3,3, 1,1},{2,2,3,4, 2,1},

        // V=2, d=3 (s_close), fi=1
        {2,3,1,1, 1,1},{2,3,1,2, 1,1},{2,3,1,3, 1,1},{2,3,1,4, 2,1},
        // V=2, d=3, fi=2
        {2,3,2,1, 2,1},{2,3,2,2, 2,1},{2,3,2,3, 2,1},{2,3,2,4, 2,1},
        // V=2, d=3, fi=3
        {2,3,3,1, 2,1},{2,3,3,2, 2,1},{2,3,3,3, 2,1},{2,3,3,4, 2,1},
    }};
};