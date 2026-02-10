#include <stdlib.h>

int main() {
    system("python3.11 -m venv thepages");
    system("source thepages/bin/activate");
    system("pip install -r requirements.txt");
    system("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118");
    return 0;
}