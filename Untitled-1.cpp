#include <iostream>
#include <unistd.h>
#include <string.h>
using namespace std;

int main() {
    int fd[2];
    pipe(fd);

    pid_t pid = fork();

    if (pid > 0) { // Parent
        close(fd[0]); // close read end
        char msg[] = "Hello Child";
        write(fd[1], msg, strlen(msg) + 1);
        close(fd[1]);
    } else { // Child
        close(fd[1]); // close write end
        char buffer[100];
        read(fd[0], buffer, sizeof(buffer));
        cout << "Child received: " << buffer << endl;
        close(fd[0]);
    }

    return 0;
}