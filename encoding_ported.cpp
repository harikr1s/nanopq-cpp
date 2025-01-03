// attempt to port the encoding feature of nanopq over to cpp

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

void txt_to_2dvec(std::fstream &file, std::vector<std::vector<float>> &vec)
{
    std::string line;
    int i{};

    while (std::getline(file, line))
    {
        float value;
        std::stringstream ss(line);

        vec.push_back(std::vector<float>());

        while (ss >> value)
        {
            vec[i].push_back(value);
        }
        ++i;
    }
    file.close();
}

int main()
{

    // reading required variables from from .py as .txt

    std::ifstream variables("outs/variables.txt");
    int Ds, M, codewords_x, codewords_y, codewords_z;
    bool verbose;
    variables >> Ds;
    variables >> M;
    variables >> verbose;

    variables >> codewords_x;
    variables >> codewords_y;
    variables >> codewords_z;

    variables.close();

    // loading 'vecs' from a .txt into a 2d <vector>

    std::vector<std::vector<float>> vecs;
    std::fstream vecs_file("outs/vecs.txt");
    if (vecs_file.is_open())
    {
        txt_to_2dvec(vecs_file, vecs);
    }

    // loading 'vecs_sub' from a .txt into a 2d vector

    std::vector<std::vector<float>> vecs_sub;
    std::fstream vecs_sub_file("outs/vecs_sub.txt");
    if (vecs_sub_file.is_open())
    {
        txt_to_2dvec(vecs_sub_file, vecs_sub);
    }

    // loading 'codes' from a .txt into a 2d vector

    std::vector<std::vector<float>> codes;
    std::fstream codes_file("outs/codes.txt");
    if (codes_file.is_open())
    {
        txt_to_2dvec(codes_file, codes);
    }

    // loading codewords as a reshaped 2d .txt into 3d vector

    std::ifstream codewords_file("outs/codewords.txt");
    std::string codewords_line;
    std::vector<std::vector<std::vector<float>>> codewords(codewords_x, std::vector<std::vector<float>>(codewords_y, std::vector<float>(codewords_z, 0)));
    int i{}, j{}, k{};
    while (std::getline(codewords_file, codewords_line) && i < codewords_x)
    {
        std::stringstream ss(codewords_line);
        std::string value;
        j = 0;
        while (std::getline(ss, value, ' ') && j * k < codewords_y * codewords_z)
        {
            codewords[i][j][k] = stof(value);
            if (k < codewords_z - 1)
            {
                k++;
            }
            else
            {
                j++, k = 0;
            }
        }
        i++;
    }

    // encoding, i.e. vector quantization

    for (int m{}; m < M; m++)
    {
        if (verbose == true)
        {
            std::cout << "Encoding the subspace: " << m + 1 << " / " << M << std::endl;
        }

        // slicing from index m*Ds to (m+1)*Ds for all rows of vecs into vecs_sub

        for (int i{}; i < vecs_sub.size(); i++)
        {
            for (int j{}, k{}; j < vecs_sub[i].size(); j++, k++)
            {
                vecs_sub[i][j] = vecs[i][(m * Ds) + k];
            }
        }

        for (int i{}; i < vecs_sub.size(); i++)
        {
            float min_dist{std::numeric_limits<float>::max()}, euclidean{};
            int id{};
            for (int j{}; j < codewords[m].size(); j++)
            {
                float squared_dist{};
                for (int k{}; k < vecs_sub[i].size(); k++)
                {
                    squared_dist += pow((vecs_sub[i][k] - codewords[m][j][k]), 2);
                }
                euclidean = sqrt(squared_dist);
                if (euclidean < min_dist)
                {
                    min_dist = euclidean;
                    id = j;
                }
            }
            codes[i][m] = id;
        }
    }

    std::ofstream file("outs/cpp_encoded.txt");
    for (int i{}; i < codes.size(); i++)
    {
        for (int j{}; j < codes[i].size(); j++)
        {
            file << codes[i][j];
            if (j != codes[i].size() - 1)
            {
                file << " ";
            }
        }
        file << "\n";
    }
    file.close();

    // for debugging

    // for (int x{}; x < codes.size(); x++)
    // {
    //     for (int y{0}; y < codes[x].size(); y++)
    //     {
    //         std::cout << codes[x][y] << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}