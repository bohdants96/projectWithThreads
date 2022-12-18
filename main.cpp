#include <iostream>
#include <fstream>
#include <string>
#include<chrono>
#include<vector>
#include <sstream>

#ifdef _WINDOWS
#include<tchar.h>
#include<strsafe.h>
#include <Windows.h>
#elif __linux__
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <pthread.h>
#include <unistd.h>
#include <cstring>
#include <semaphore.h>
#endif

//defining
#define CUR_THREADS 8

typedef long long ll;

using namespace std;


ll counter = 0;
#ifdef _WINDOWS
HANDLE Semaphore;
LONG Max = 2;
HANDLE Mutex;
CRITICAL_SECTION criticalSection;
ll timer = 1000;
#elif __linux__
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
sem_t sem;
int signal = 0;
ll Max = 2;
ll timer = 100;

struct FileMapping {
    int fd;
    size_t fsize;
    char* dataPtr;
};
#endif

vector<pair<chrono::time_point<chrono::steady_clock>, chrono::time_point<chrono::steady_clock>>>
        times;


//checking are the files equal
bool eqFiles(string file1, string file2) {
    ifstream f1(file1.c_str(), ios::binary);
    ifstream f2(file2.c_str(), ios::binary);
    if (f1.is_open() && f2.is_open()) {
        //checking are the numbers of char equal
        f1.seekg(0, ios::end);
        f2.seekg(0, ios::end);
        if (f1.tellg() != f2.tellg()) {
            f1.close();
            f2.close();
            return false;
        }
        //checking are the characters equal
        f1.seekg(0, ios::beg);
        f2.seekg(0, ios::beg);
        char c1, c2;
        while (f1.get(c1) && f2.get(c2)) {
            if (c1 != c2) {
                f1.close();
                f2.close();
                return false;
            }
        }
        f1.close();
        f2.close();
        return true;
    } else {
        return false;
    }
}

typedef struct DataForMulti {

    ll *matrix1; //first matrix
    ll *matrix2; //second matrix
    ll *res_matrix; //matrix-result
    chrono::time_point<chrono::steady_clock> start_program;//start time for program
    int size_matrix; //size size_matrix*size_matrix
    int pos_thread;  //start pos for thread
    int block_size;  //block size for thread
    int t;
} THREAD, *PTHREAD;

#ifdef _WINDOWS
DWORD WINAPI MyThreadFunction(LPVOID lpParam) {
    BOOL bContinue = TRUE;

    auto *data = (PTHREAD) lpParam;

    ll *m1 = data->matrix1;
    ll *m2 = data->matrix2;
    ll *res = data->res_matrix;
    ll n = data->size_matrix;
    ll pos_thread = data->pos_thread;
    ll block_size = data->block_size;
    ll t = data->t;

    chrono::time_point<chrono::steady_clock> start_program = data->start_program;
    while (bContinue) {
        DWORD dwWaitResult = WaitForSingleObject(
                Semaphore,   // handle to semaphore
                INFINITE);           // zero-second time-out interval

        switch (dwWaitResult) {
// The semaphore object was signaled.
            case WAIT_OBJECT_0:
                for (
                        ll x = pos_thread;
                        x < (block_size + pos_thread); x++) {

                    ll i = x / n;
                    ll j = x % n;

                    res[i * n + j] = 0;

                    for (ll k = 0;k < n;k++) {
                        res[i * n+ j] += m1[i * n+ k] * m2[k * n+ j];
                    }

                    EnterCriticalSection(&criticalSection);
                    counter++;
                    chrono::time_point<chrono::steady_clock> now = chrono::steady_clock::now();
                    if (chrono::duration_cast<chrono::milliseconds>(now- start_program).count()> timer) {
                        timer += 1000;
                        long double proc = ((long double) counter / (long double) (n * n)) * (long double) 100;
                        cout << "Calculating: " << proc << "%" <<endl;
                    }
                    LeaveCriticalSection(&criticalSection);
                    /*  DWORD dwWaitResult1 = WaitForSingleObject(Mutex, INFINITE);
                      switch (dwWaitResult1) {
                          case WAIT_OBJECT_0:
                              counter++;
                              chrono::time_point<chrono::steady_clock> now = chrono::steady_clock::now();
                              if (chrono::duration_cast<chrono::milliseconds>(now- start_program).count()> timer) {
                                  timer += 1000;
                                  long double proc = ((long double) counter / (long double) (n * n)) * (long double) 100;
                                  cout << "Calculating: " << proc << "%" <<endl;
                              }
                      }
                      if (!ReleaseMutex(Mutex)) {
                          printf("ReleaseSemaphore error: %d\n",GetLastError());
                      }*/
                    bContinue = FALSE;
                }
                times[t].second = chrono::steady_clock::now();

                HeapFree(GetProcessHeap(),0, lpParam);
// Release the semaphore when task is finished

                if (!ReleaseSemaphore(
                        Semaphore,  // handle to semaphore
                        1,            // increase count by one
                        NULL))       // not interested in previous count
                {
                    printf("ReleaseSemaphore error: %d\n",GetLastError());
                }
                break;

// The semaphore was nonsignaled, so a time-out occurred.
            case WAIT_TIMEOUT:
                printf("Thread %d: wait timed out\n",GetCurrentThreadId());
                break;
        }
    }
    return TRUE;
}
#elif __linux__
void* MyThreadFunction(void* arg){
    auto *data = (PTHREAD) arg;

    ll t = data->t;
    ll *m1 = data->matrix1;
    ll *m2 = data->matrix2;
    ll *res = data->res_matrix;
    ll n = data->size_matrix;
    ll pos_thread = data->pos_thread;
    ll block_size = data->block_size;
    chrono::time_point<chrono::steady_clock> start_program = data->start_program;

    sem_wait(&sem);
//calculating block of matrix-result
    for (ll x = pos_thread; x < (block_size + pos_thread); x++) {
        ll i = x / n;
        ll j = x % n;

        res[i * n + j] = 0;

        for (ll k = 0; k < n; k++) {
            res[i * n + j] += m1[i * n + k] * m2[k * n + j];
        }

        pthread_mutex_lock(&mutex);
        // pthread_cond_init(&cond, NULL);
        counter++;

        chrono::time_point<chrono::steady_clock> now = chrono::steady_clock::now();
        if (chrono::duration_cast<chrono::milliseconds>(now- start_program).count()> timer) {
            timer += 100;
            long double proc = ((long double) counter / (long double) (n * n)) * (long double) 100;
            cout << "Calculating: " << proc << "%" <<endl;
        }
        pthread_mutex_unlock(&mutex);
        //pthread_cond_destroy(&cond);
    }
    times[t].second = chrono::steady_clock::now();
    sem_post(&sem);

    return nullptr;
}
#endif

long long *multi(ll *matrix1, ll *matrix2, int size_matrix) {
    ll *res_matrix = new ll[size_matrix * size_matrix];


#ifdef _WINDOWS
    DWORD dwThreadIndetifire[CUR_THREADS];
    HANDLE hThreadArray[CUR_THREADS];
    PTHREAD pInfos[CUR_THREADS];

    Semaphore = CreateSemaphore(
            NULL,
            Max,
            Max,
            NULL
    );

    if (Semaphore == NULL) {
        cout << "Creating Semaphore is failed" << endl;
        ExitProcess(3);
    }
    Mutex = CreateMutex(
            NULL,
            FALSE,
            "Mutex"
    );
    if (Mutex == NULL) {
        cout << "Creating Mutex is failed" << endl;
        ExitProcess(3);
    }
    InitializeCriticalSection(&criticalSection);

#elif __linux__
    pthread_t id[CUR_THREADS];
    THREAD pInfos[CUR_THREADS];
    pthread_attr_t thAttr[CUR_THREADS];

    if ( sem_init(&sem, 0, Max) != 0 )
    {
        cout<<"Semaphore error"<<endl;
        exit(3);
    }
#endif

    times.resize(CUR_THREADS);


    //current position, block size for matrix and mod
    int default_block_size = (size_matrix * size_matrix) / CUR_THREADS;
    int mod = (size_matrix * size_matrix) % CUR_THREADS;
    int current_block = 0;
    chrono::time_point<chrono::steady_clock> begin_time = chrono::steady_clock::now();
    //creating threads for matrix

    for (int i = 0; i < CUR_THREADS; i++) {
#ifdef _WINDOWS
        pInfos[i] = (PTHREAD) HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(THREAD));

        if (pInfos[i] == nullptr) {
            cout << "HeapAlloc is failed" << endl;
            ExitProcess(2);
        }

        times[i].first = begin_time;
        pInfos[i]->matrix1 = matrix1;
        pInfos[i]->matrix2 = matrix2;
        pInfos[i]->res_matrix = res_matrix;
        pInfos[i]->size_matrix = size_matrix;
        pInfos[i]->pos_thread = current_block;
        pInfos[i]->block_size = default_block_size;
        pInfos[i]->start_program = begin_time;
        pInfos[i]->t = i;

        if (mod > 0) {
            pInfos[i]->block_size++;
            mod--;
        }
        current_block += pInfos[i]->block_size;
#elif __linux__
        times[i].first = begin_time;
        pInfos[i].matrix1 = matrix1;
        pInfos[i].matrix2 = matrix2;
        pInfos[i].res_matrix = res_matrix;
        pInfos[i].size_matrix = size_matrix;
        pInfos[i].pos_thread = current_block;
        pInfos[i].block_size = default_block_size;
        pInfos[i].start_program = begin_time;
        pInfos[i].t = i;

        if (mod > 0) {
            pInfos[i].block_size++;
            mod--;
        }
        current_block += pInfos[i].block_size;
#endif
#ifdef _WINDOWS
        hThreadArray[i] = CreateThread(
                nullptr,               // default security attributes
                0,                      // use default stack size
                MyThreadFunction,                        // thread function name
                pInfos[i],                     // argument to thread function
                0,                    // use default creation flags
                &dwThreadIndetifire[i]                 // returns the thread identifier
        );
        if (hThreadArray[i] == nullptr) {
            cout << "CreateThread is failed" << endl;
            ExitProcess(3);
        }


    }
    for (ll i = 0; i < CUR_THREADS; i++) {
        WaitForSingleObject(hThreadArray[i], INFINITE);
    }

    for (auto &i: hThreadArray) {
        CloseHandle(i);
    }
    CloseHandle(Semaphore);
    CloseHandle(Mutex);
    DeleteCriticalSection(&criticalSection);

#elif __linux__
    pthread_create(&id[i], NULL, MyThreadFunction, &pInfos[i]);

    }
    for (int i = 0; i < CUR_THREADS; i++)
    {
        pthread_join(id[i], NULL);
    }
#endif

    return res_matrix;
}

void solution(ll n,ll *matrix1,ll*matrix2,ll *result, string output) {

    //integer for timer

    cout << "Calculating " << n << "*" << n << " matrix..." << endl;
    chrono::time_point<chrono::steady_clock> beginW = chrono::steady_clock::now();
    //calculating our result matrix
    result = multi(matrix1, matrix2, n);

    chrono::time_point<chrono::steady_clock> endW = chrono::steady_clock::now();
    //timer
    for(ll i =0;i<CUR_THREADS;i++){
        cout<<"Thread["<<i<<"]: "<<chrono::duration_cast<chrono::milliseconds>(times[i].second-times[i].first).count()<<" ms "<<endl;
    }
    cout << "Time for " << n << "*" << n << ": " << chrono::duration_cast<chrono::milliseconds>(endW-beginW).count()<< " ms" << endl;


    //input input file
    //freopen(output.c_str(), "w", stdout);

    fstream fout;
    fout.open(output,ios_base::out);
    if(!fout.is_open()){
        cout<<"Error open file"<<endl;
    }else {
        for (ll i = 0; i < n; i++) {
            for (ll j = 0; j < n; j++) {
                fout<< result[i * n + j]<< " ";
            }
            fout<<"\n";
        }

    }
    fout.close();


}

void checkingAns(string s1, string s2, string s3) {
    if (!eqFiles(s1, s2)) {
        cout << s3 << " checkingAns is failed" << endl;
    } else {
        cout << s3 << " checkingAns is passed" << endl;
    }
}

#ifdef _WINDOWS
void fileMapping(){
    HANDLE hFile;
    DWORD dwFileSize;
    HANDLE hFileMapping;
    char *line;
    BOOL bResult;
    hFile = CreateFile(
            (LPCTSTR) "C:\\Test\\test_big.txt",
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        cout << "CreateFile error: " << GetLastError() << endl;
        exit(4);
    }
    cout << "CreateFile success" << endl;

    dwFileSize = GetFileSize(hFile, NULL);
    if (dwFileSize == INVALID_FILE_SIZE) {
        cout << "FileSize error: " << GetLastError() << endl;
        CloseHandle(hFile);
        exit(5);
    }
    cout << "GetFileSize success" << endl;

    hFileMapping = CreateFileMapping(
            hFile,
            NULL,
            PAGE_READONLY,
            0,
            dwFileSize,
            NULL
    );

    if (hFileMapping == NULL) {
        cout << "CreateFileMapping error: " << GetLastError() << endl;
        CloseHandle(hFile);
        exit(6);
    }
    cout << "CreateFileMapping success" << endl;

    line = (char *) MapViewOfFile(
            hFileMapping,
            FILE_MAP_READ,
            0,
            0,
            dwFileSize
    );
    if (line == NULL) {
        cout << "MapViewOfFile error: " << GetLastError() << endl;
        CloseHandle(hFile);
        CloseHandle(hFileMapping);
        exit(7);
    }
    cout << "MapViewOfFile success" << endl;

    ll i = 0;
    ll counterOfEl = 0;
    stringstream s(line);
    ll n;
    s >> n;
    ll *matrix1 = new ll[n * n];
    ll *matrix2 = new ll[n * n];
    ll *result = new ll[n * n];
    cout << "Start reading" << endl;
    while (i < n * n) {
        s >> matrix1[i];
        i++;
    }
    cout << "Matrix1 is read" << endl;

    s >> n;
    i = 0;
    while (i < n * n) {
        s >> matrix2[i];
        i++;
    }
    cout << "Matrix2 is read" << endl;

    solution(n, matrix1, matrix2, result, "C:\\Test\\out_big.txt");


    CloseHandle(hFile);
    CloseHandle(hFileMapping);
    bResult = UnmapViewOfFile(line);
    if (bResult == FALSE) {
        cout << "UnmapViewOfFile error: " << GetLastError() << endl;
    }
}
#elif __linux__
void fileMapping(){
    const char *fname = "/home/bohdan/CLionProjects/testThreads/test_big.txt";
    int fd = open(fname, O_RDONLY, 0);
    if(fd < 0) {
        std::cerr << "fileMappingCreate - open failed, fname =  "
                  << fname << ", " << strerror(errno) << std::endl;
        exit(4);
    }
    cout << "Open file success" << endl;

    struct stat st;
    if(fstat(fd, &st) < 0) {
        std::cerr << "fileMappingCreate - fstat failed, fname = "
                  << fname << ", " << strerror(errno) << std::endl;
        close(fd);
        exit(5);
    }
    size_t fsize = (size_t)st.st_size;
    cout<<"Get size success"<<endl;
    char* dataPtr = (char*)mmap(nullptr, fsize,
                                PROT_READ,
                                MAP_PRIVATE,
                                fd, 0);
    if(dataPtr == MAP_FAILED) {
        std::cerr << "fileMappingCreate - mmap failed, fname = "
                  << fname << ", " << strerror(errno) << std::endl;
        close(fd);
        exit(7);
    }
    cout<<"Mmap success"<<endl;

    FileMapping * mapping = (FileMapping *)malloc(sizeof(FileMapping));
    if(mapping == nullptr) {
        std::cerr << "fileMappingCreate - malloc failed, fname = "
                  << fname << std::endl;
        munmap(dataPtr, fsize);
        close(fd);
        exit(8);
    }

    mapping->fd = fd;
    mapping->fsize = fsize;
    mapping->dataPtr = dataPtr;
    cout<<"Getting mapping success"<<endl;

    ll i = 0;
    ll counterOfEl = 0;
    stringstream s(dataPtr);
    ll n;
    s >> n;
    ll *matrix1 = new ll[n * n];
    ll *matrix2 = new ll[n * n];
    ll *result = new ll[n * n];
    cout << "Start reading" << endl;
    while (i < n * n) {
        s >> matrix1[i];
        i++;
    }
    cout << "Matrix1 is read" << endl;

    s >> n;
    i = 0;
    while (i < n * n) {
        s >> matrix2[i];
        i++;
    }
    cout << "Matrix2 is read" << endl;

    solution(n, matrix1, matrix2, result, "/home/bohdan/CLionProjects/testThreads/out_big.txt");

    munmap(mapping->dataPtr, mapping->fsize);
    close(mapping->fd);
    free(mapping);
}
#endif

int main(int argc, char **argv) {


    cout << endl << "Threads " << CUR_THREADS << ":" << endl << endl;
    cout << "Threads simultaneously: " << Max << endl << endl;

    fileMapping();
#ifdef _WINDOWS
    checkingAns("C:\\Test\\out_big.txt", "C:\\Test\\ans_big.txt", "Big");
#elif __linux__
    checkingAns("/home/bohdan/CLionProjects/testThreads/out_big.txt", "/home/bohdan/CLionProjects/testThreads/ans_big.txt", "Big");
#endif
    cout << endl;

    cout << "Program is done" << endl;

    return 0;
}
