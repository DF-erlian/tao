#include<iostream>
#include<sstream>
#include<fstream>
#include<string>
#include<map>

using namespace std;

const char* workload = "/worksapce/TAO/MLSim/workload/spec2006/bzip2/trace_o3_train.txt";

int main(){
    ifstream data(workload);
    if(!data.is_open()){
        cout << "cannot open workload, check the filepath\n";
        return 0;
    }

    map<uint64_t, long long> fetchTick;

    uint64_t preTick = -1;
    bool first = true;
    string line;
    uint64_t curTick;
    while(getline(data, line)){
        stringstream ss(line);
        ss >> curTick;

        if(first){
            preTick = curTick;
            first = false;
        }
        else{
            fetchTick[(curTick - preTick)/250] += 1;
            preTick = curTick;
        }
    }

    for(auto& [fetchtick, times] : fetchTick){
        cout << fetchtick << " appears " << times << " times.\n";
    }
    return 0;
}