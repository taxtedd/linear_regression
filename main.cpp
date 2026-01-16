#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

namespace Metrics {
    double mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        if (y_true.empty()) return 0.0;
        double mse = 0.0;
        for (size_t i = 0; i < y_true.size(); i++) {
            double err = y_true[i] - y_pred[i];
            mse += err * err;
        }
        return mse / y_true.size();
    }

    double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        if (y_true.empty()) return 0.0;
        double mean_y = 0.0; 
        for (double val : y_true) mean_y += val;
        mean_y /= y_true.size();

        double ss_res = 0.0; 
        double ss_tot = 0.0; 

        for (size_t i = 0; i < y_true.size(); ++i) {
            ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
            ss_tot += (y_true[i] - mean_y) * (y_true[i] - mean_y);
        }
        
        if (ss_tot < 1e-9) return 0.0; 
        return 1.0 - (ss_res / ss_tot);
    }
}

class LinearRegression {
private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int iterations;

public:
    LinearRegression(double lr = 0.01, int iter = 1000) 
        : learning_rate(lr), iterations(iter), bias(0.0) {}

    double predict_one(const std::vector<double>& features) const {
        double prediction = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            prediction += weights[i] * features[i];
        }
        return prediction;
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<double> predictions;
        predictions.reserve(X.size());
        for (const auto& row : X) {
            predictions.push_back(predict_one(row));
        }
        return predictions;
    }

    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<double>& y,
             const std::vector<std::vector<double>>& X_val = {},
             const std::vector<double>& y_val = {},
             int log_freq = 100)                                
    {
        if (X.empty()) return;
        size_t n_samples = X.size();
        size_t n_features = X[0].size();
        
        weights.assign(n_features, 0.0);
        bias = 0.0;

        std::cout << "\nStarting training..." << std::endl;
        std::cout << std::setw(6) << "Epoch" 
                  << " | " << std::setw(10) << "Train MSE" 
                  << " | " << std::setw(10) << "Train R2";
        
        if (!X_val.empty()) {
            std::cout << " | " << std::setw(10) << "Val MSE" 
                      << " | " << std::setw(10) << "Val R2";
        }
        std::cout << "\n----------------------------------------------------------------" << std::endl;

        for (int iter = 0; iter < iterations; iter++) {
            // Градиентный спуск
            std::vector<double> dw(n_features, 0.0);
            double db = 0.0;

            for (size_t i = 0; i < n_samples; i++) {
                double error = predict_one(X[i]) - y[i];
                for (size_t j = 0; j < n_features; ++j) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            for (size_t j = 0; j < n_features; ++j) {
                weights[j] -= learning_rate * (2.0 / n_samples) * dw[j];
            }
            bias -= learning_rate * (2.0 / n_samples) * db;

            // Метрики
            if (iter == 0 || (iter + 1) % log_freq == 0 || iter == iterations - 1) {
                auto train_preds = predict(X);
                double train_mse = Metrics::mean_squared_error(y, train_preds);
                double train_r2 = Metrics::r2_score(y, train_preds);

                std::cout << std::setw(6) << (iter + 1)
                          << " | " << std::fixed << std::setprecision(4) << std::setw(10) << train_mse
                          << " | " << std::fixed << std::setprecision(4) << std::setw(10) << train_r2;

                if (!X_val.empty()) {
                    auto val_preds = predict(X_val);
                    double val_mse = Metrics::mean_squared_error(y_val, val_preds);
                    double val_r2 = Metrics::r2_score(y_val, val_preds);

                    std::cout << " | " << std::setw(10) << val_mse
                              << " | " << std::setw(10) << val_r2;
                }
                std::cout << std::endl;
            }
        }
        std::cout << "----------------------------------------------------------------\n" << std::endl;
    }
};

class StandardScaler {
private:
    std::vector<double> mean;
    std::vector<double> scale; // std

public:
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.empty()) return;
        size_t n_samples = X.size();
        size_t n_features = X[0].size(); 

        mean.assign(n_features, 0.0);
        scale.assign(n_features, 0.0);

        for (const auto& row : X) {
            for (size_t j = 0; j < n_features; ++j) {
                mean[j] += row[j];
            }
        }
        for (size_t j = 0; j < n_features; ++j) mean[j] /= n_samples;

        for (const auto& row : X) {
            for (size_t j = 0; j < n_features; ++j) {
                scale[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
            }
        }
        for (size_t j = 0; j < n_features; ++j) {
            scale[j] = std::sqrt(scale[j] / n_samples);
            if (scale[j] < 1e-9) scale[j] = 1.0; // от деления на 0
        }
    }

    void transform(std::vector<std::vector<double>>& X) {
        for (auto& row : X) {
            for (size_t j = 0; j < row.size(); ++j) {
                row[j] = (row[j] - mean[j]) / scale[j];
            }
        }
    }
};

void train_val_test_split(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::vector<std::vector<double>>& X_train,
    std::vector<double>& y_train,
    std::vector<std::vector<double>>& X_val,
    std::vector<double>& y_val,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_test,
    double val_size = 0.15,
    double test_size = 0.15
) {
    size_t n = X.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0); // заполнение индексами

    std::mt19937 g(42);
    std::shuffle(idx.begin(), idx.end(), g);

    size_t n_test = n * test_size;
    size_t n_val  = n * val_size;

    for (size_t i = 0; i < n; ++i) {
        size_t id = idx[i];
        if (i < n_test) {
            X_test.push_back(X[id]);
            y_test.push_back(y[id]);
        } else if (i < n_test + n_val) {
            X_val.push_back(X[id]);
            y_val.push_back(y[id]);
        } else {
            X_train.push_back(X[id]);
            y_train.push_back(y[id]);
        }
    }
}

void generate_synthetic_data(size_t samples, std::vector<std::vector<double>>& X, std::vector<double>& y) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> d_x1(0.0, 10.0);
    std::uniform_real_distribution<> d_x2(-5.0, 5.0);
    std::uniform_real_distribution<> d_x3(100.0, 200.0);
    std::normal_distribution<> d_noise(0.0, 5.0);

    for (size_t i = 0; i < samples; i++) {
        double x1 = d_x1(gen);
        double x2 = d_x2(gen);
        double x3 = d_x3(gen);
        
        std::vector<double> row = {x1, x2, x3};
        X.push_back(row);
        y.push_back(3.5 * x1 - 2.0 * x2 + 1.5 * x3 + 10.0 + d_noise(gen));
    }
}

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    int n = 10000;
    generate_synthetic_data(n, X, y);

    std::vector<std::vector<double>> X_train, X_val, X_test;
    std::vector<double> y_train, y_val, y_test;
    double val_size = 0.1;
    double test_size = 0.2;
    train_val_test_split(X, y,
                     X_train, y_train,
                     X_val, y_val,
                     X_test, y_test,
                     val_size, test_size);

    StandardScaler scaler;
    scaler.fit(X_train);
    scaler.transform(X_train);
    scaler.transform(X_val);
    scaler.transform(X_test);

    double lr = 0.1;
    int iterations = 1000;
    LinearRegression model(lr, iterations);
    
    int log_freq = 50;
    model.fit(X_train, y_train, X_val, y_val, log_freq);

    auto test_preds = model.predict(X_test);

    double test_mse = Metrics::mean_squared_error(y_test, test_preds);
    double test_r2  = Metrics::r2_score(y_test, test_preds);
    std::cout << "\nTest metrics:\n";
    std::cout << "MSE = " << test_mse << "\n";
    std::cout << "R2  = " << test_r2 << "\n";
}