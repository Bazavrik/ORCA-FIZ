#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

class FuzzySD {
public:
    explicit FuzzySD(std::size_t grid_n = 301) : grid_n_(grid_n < 2 ? 2 : grid_n) {}

    double eval(double V, double d, double fi, double C) const {
        V  = clamp(V,  0.0, 30.0);
        d  = clamp(d,  0.0, 20.0);
        fi = clamp(fi, 0.0, 180.0);
        C  = clamp(C,  0.0, 4.0);

        const double muV_same      = trapmf(V,  -11.25, 0.0,  2.0, 10.0);
        const double muV_diff      = trapmf(V,   5.0,  15.0, 30.0, 31.0);
        const std::array<double, 2> muV{muV_same, muV_diff};

        const double muD_close     = trapmf(d,   1.0,  3.0,  6.0,  8.0);
        const double muD_far       = trapmf(d,   5.0, 10.0, 20.0, 21.0);
        const double muD_sclose    = trimf(d,   -1.0,  0.0,  4.0);
        const std::array<double, 3> muD{muD_close, muD_far, muD_sclose};

        const double muFi_same     = trapmf(fi, -1.0,  0.0, 10.0, 50.0);
        const double muFi_perp     = trapmf(fi, 40.0, 75.0,105.0,135.0);
        const double muFi_opp      = trapmf(fi,115.0,150.0,180.0,190.0);
        const std::array<double, 3> muFi{muFi_same, muFi_perp, muFi_opp};

        const double muC_animal    = trimf(C,   2.0, 3.0, 4.0);
        const double muC_bicycle   = trimf(C,   0.0, 1.0, 2.0);
        const double muC_human     = trimf(C,   1.0, 2.0, 3.0);
        const double muC_undef     = trimf(C,   3.0, 4.0, 5.0);
        const std::array<double, 4> muC{muC_animal, muC_bicycle, muC_human, muC_undef};

        double alpha_safe   = 0.0;
        double alpha_danger = 0.0;

        for (const auto& r : rules_) {
            const double firing = std::min(
                std::min(muV[r.iV - 1], muD[r.iD - 1]),
                std::min(muFi[r.iFi - 1], muC[r.iC - 1])
            ) * r.weight;

            if (r.out == 1) alpha_safe   = std::max(alpha_safe,   firing);
            else            alpha_danger = std::max(alpha_danger, firing);
        }

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

        if (den <= 1e-12) return 0.5;
        return clamp(num / den, 0.0, 1.0);
    }

    std::vector<double> eval_batch_flat(const double* x, std::size_t rows, std::size_t cols) const {
        std::vector<double> out;
        if (cols != 4) {
            return out;
        }
        out.reserve(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            const double* row = x + i * cols;
            out.push_back(eval(row[0], row[1], row[2], row[3]));
        }
        return out;
    }

private:
    struct Rule {
        int iV;
        int iD;
        int iFi;
        int iC;
        int out;
        double weight;
    };

    std::size_t grid_n_;

    static double clamp(double x, double a, double b) {
        return std::max(a, std::min(b, x));
    }

    static double trapmf(double x, double a, double b, double c, double d) {
        if (x <= a || x >= d) return 0.0;
        if (x >= b && x <= c) return 1.0;
        if (x > a && x < b)   return (x - a) / (b - a);
        return (d - x) / (d - c);
    }

    static double trimf(double x, double a, double b, double c) {
        if (x <= a || x >= c) return 0.0;
        if (x == b) return 1.0;
        if (x > a && x < b) return (x - a) / (b - a);
        return (c - x) / (c - b);
    }

    static constexpr std::array<Rule, 72> rules_ = {{
        {1,1,1,1, 2,1},{1,1,1,2, 1,1},{1,1,1,3, 1,1},{1,1,1,4, 2,1},
        {1,1,2,1, 2,1},{1,1,2,2, 1,1},{1,1,2,3, 2,1},{1,1,2,4, 2,1},
        {1,1,3,1, 2,1},{1,1,3,2, 2,1},{1,1,3,3, 2,1},{1,1,3,4, 2,1},
        {1,2,1,1, 1,1},{1,2,1,2, 1,1},{1,2,1,3, 1,1},{1,2,1,4, 1,1},
        {1,2,2,1, 1,1},{1,2,2,2, 1,1},{1,2,2,3, 1,1},{1,2,2,4, 1,1},
        {1,2,3,1, 1,1},{1,2,3,2, 1,1},{1,2,3,3, 1,1},{1,2,3,4, 2,1},
        {1,3,1,1, 2,1},{1,3,1,2, 2,1},{1,3,1,3, 2,1},{1,3,1,4, 2,1},
        {1,3,2,1, 2,1},{1,3,2,2, 2,1},{1,3,2,3, 2,1},{1,3,2,4, 2,1},
        {1,3,3,1, 2,1},{1,3,3,2, 2,1},{1,3,3,3, 2,1},{1,3,3,4, 2,1},
        {2,1,1,1, 2,1},{2,1,1,2, 1,1},{2,1,1,3, 1,1},{2,1,1,4, 2,1},
        {2,1,2,1, 2,1},{2,1,2,2, 1,1},{2,1,2,3, 1,1},{2,1,2,4, 2,1},
        {2,1,3,1, 2,1},{2,1,3,2, 2,1},{2,1,3,3, 2,1},{2,1,3,4, 2,1},
        {2,2,1,1, 1,1},{2,2,1,2, 1,1},{2,2,1,3, 1,1},{2,2,1,4, 1,1},
        {2,2,2,1, 1,1},{2,2,2,2, 1,1},{2,2,2,3, 1,1},{2,2,2,4, 1,1},
        {2,2,3,1, 2,1},{2,2,3,2, 1,1},{2,2,3,3, 1,1},{2,2,3,4, 2,1},
        {2,3,1,1, 1,1},{2,3,1,2, 1,1},{2,3,1,3, 1,1},{2,3,1,4, 2,1},
        {2,3,2,1, 2,1},{2,3,2,2, 2,1},{2,3,2,3, 2,1},{2,3,2,4, 2,1},
        {2,3,3,1, 2,1},{2,3,3,2, 2,1},{2,3,3,3, 2,1},{2,3,3,4, 2,1},
    }};
};
