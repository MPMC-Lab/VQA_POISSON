// IOmethods.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../VQA.hpp"


void split_tokens(const std::string& line, std::vector<std::string>& out) {
    std::string token;
    for (char c : line) {
        if (c == ',' || c == ' ' || c == '\t' || c == ';')
            c = ' ';
        token.push_back(c);
    }
    std::istringstream iss(token);
    std::string w;
    while (iss >> w) out.push_back(w);
}

std::vector<double> load_params_csv(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("cannot open " + path);

    std::vector<std::string> toks;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        split_tokens(line, toks);
    }
    std::vector<double> vals;
    vals.reserve(toks.size());
    for (auto& s : toks) vals.push_back(std::stod(s));
    return vals;
}