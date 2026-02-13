#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: ./wavmaker_fast <folder> <output_name> <wav|mp3>\n";
        return 1;
    }

    std::string folder = argv[1];
    std::string output = argv[2];
    std::string type   = argv[3];

    if (type != "wav" && type != "mp3") {
        std::cout << "Error: Output type must be 'wav' or 'mp3'\n";
        return 1;
    }

    std::vector<std::string> files;

    std::cout << "[DEBUG] Scanning: " << folder << std::endl;

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".wav") {
            files.push_back(entry.path().filename().string());
        }
    }

    if (files.empty()) {
        std::cout << "No WAV files found.\n";
        return 1;
    }

    std::sort(files.begin(), files.end());
    std::cout << "[DEBUG] Found " << files.size() << " WAV files.\n";

    // Create concat list
    std::ofstream list_file("files.txt");
    for (const auto& f : files) {
        list_file << "file '" << folder << "/" << f << "'\n";
    }
    list_file.close();

    int result = 0;

    if (type == "wav") {
        std::string cmd =
            "ffmpeg -f concat -safe 0 -i files.txt -c copy " +
            output + ".wav -y";

        std::cout << "[DEBUG] Creating combined WAV...\n";
        result = std::system(cmd.c_str());
    }
    else if (type == "mp3") {
        std::string temp_wav = output + "_temp.wav";

        std::string concat_cmd =
            "ffmpeg -f concat -safe 0 -i files.txt -c copy " +
            temp_wav + " -y";

        std::cout << "[DEBUG] Creating temporary WAV...\n";
        result = std::system(concat_cmd.c_str());

        if (result != 0) return result;

        std::string encode_cmd =
            "ffmpeg -i " + temp_wav +
            " -c:a libmp3lame -q:a 3 -compression_level 0 " +
            output + ".mp3 -y";

        std::cout << "[DEBUG] Encoding MP3...\n";
        result = std::system(encode_cmd.c_str());from zipvoice.luxvoice import LuxTTS, get_memory_info, clear_memory, unload_model

        // Clean up temp file
        std::filesystem::remove(temp_wav);
    }

    return result;
}
