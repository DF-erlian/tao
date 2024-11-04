#include<iostream>
#include "inst.h"
#include<fstream>
#include<queue>
#include<unordered_map>
#include<map>

using namespace std;

const char* workload = "../workload/spec2006/bzip2/trace_o3_train.txt";
const char* trainData = "../workload/spec2006/bzip2/trainData/data.txt";

const int Num_OpClass = 74;
const int TicksPreClock = 250;

int main(){
    ifstream trace(workload);

    Tick curTick = -1;
    
    int numAddrHistory = 256; //超参
    vector<uint64_t> addrHistory;

    unordered_map<int, vector<int>> bpHash;
    int NumHash = 256, tableSize = 64; //超参
    for(int i = 0; i < NumHash; i++){ //初始化
        bpHash[i] = vector<int>(tableSize, 0);
    }

    map<int, int> regMap;

    string line;
    Instruction Inst;
    uint64_t lastPC = 0;

    bool first = true;

    long long instCnt = -1;
    int fileNum = 0;
    string trainDataFile = string(trainData) + to_string(fileNum);
    ofstream traindata(trainDataFile);
    if(!traindata.is_open()){
        cout << "something wrong in open new traindata file" << endl;
    }

    // //process regmap
    // while(getline(trace, line)){
    //     instCnt += 1;

    //     if(instCnt >= 10'000'000 ){
    //         break;
    //     }
    //     Inst = Instruction();
    //     Inst.read(line);

    //     for(auto&p : Inst.regVector){
    //         int idx = p.first * 1000 + p.second;
    //         if(regMap.count(idx)){
    //             continue;
    //         }
    //         else{
    //             regMap[idx] = regMap.size();
    //         }
    //     }
    // }
    // trace.close();

    // cout << regMap.size() << endl;

    // ofstream maptrace("./regMapTrace.txt");
    // for(auto [tag, idx] : regMap){
    //     maptrace << tag << "\t\t" << idx << endl;
    // }
    // maptrace.close();


    ifstream maptrace("./regMapTrace.txt");
    while(!maptrace.eof()){
        int tag, idx;
        maptrace >> tag >> idx;
        regMap[tag] = idx;
    }


    while(getline(trace, line)){
        instCnt += 1;

        if(instCnt >= 1'000'000 ){
            traindata.close();
            
            // break;
            instCnt = 0;
            fileNum += 1;
            trainDataFile = string(trainData) + to_string(fileNum);
            traindata.open(trainDataFile);

            if(!traindata.is_open()){
                cout << "something wrong in open new traindata file" << endl;
            }
        }

        if(fileNum >=10 ){
            break;
        }

        Inst = Instruction();
        Inst.read(line);

        if(first){
            curTick = Inst.fetchTick;
            first = false;
        }

        //dumpInst

        ///dump fetchTick, execTick;
        traindata << (Inst.fetchTick - curTick)/TicksPreClock << " " << Inst.execTick/TicksPreClock;
        curTick = Inst.fetchTick;

        ///dump opClass
        vector<int> opClassTable(Num_OpClass, 0);
        opClassTable[Inst.opCode] = 1;

        for(int i = 0; i < Num_OpClass; i++){
            traindata << " " << opClassTable[i];
        }

        ///dump flags
        for(int i = 0; i < Inst.NumFlags; i++){
            traindata << " " << Inst.flags[i];
        }

        ///dump registers
        vector<int> regs(regMap.size(), 0);
        for(auto&p : Inst.regVector){
            int tag = p.first * 1000 + p.second;
            if(regMap.count(tag)) regs[regMap[tag]]  = 1;
        }

        for(int t : regs){
            traindata << " " << t;
        }

        ///dump addr
        if(!Inst.isMem){
            for(int i = 0; i < numAddrHistory; i++){
                traindata << " 0";
            }
        }
        else{
            if(addrHistory.size() < numAddrHistory){
                for(int i = 0; i < numAddrHistory - addrHistory.size(); i++){
                    traindata << " 0";
                }
            }

            for(int i = 0; i < addrHistory.size(); i++){
                uint64_t accessDist = ((Inst.memAddr > addrHistory[i]) ? (Inst.memAddr - addrHistory[i]) : (addrHistory[i] - Inst.memAddr));

                if(accessDist > 1024 * 1024){
                    accessDist = 1024 * 1024 / 64 + 1;
                }
                else{
                    accessDist = accessDist / 64;
                }
                traindata << " " << accessDist;
            }

            addrHistory.push_back(Inst.memAddr);

            while(addrHistory.size() > numAddrHistory){
                addrHistory.erase(addrHistory.begin());
            }
        }

        ///dump bp
        if(!Inst.isBranch){
            for(int i = 0; i < tableSize; i++){
                traindata << " 0";
            }
        }
        else{
            int hashIdx = (Inst.pc / 4) % NumHash;
            for(int i = 0; i < tableSize; i++){
                traindata << " " << bpHash[hashIdx][i];
            }

            bpHash[hashIdx].push_back((Inst.pc - lastPC) == 4 ? 0 : 1);

            while(bpHash[hashIdx].size() > tableSize){
                bpHash[hashIdx].erase(bpHash[hashIdx].begin());
            }
        }
        traindata << endl;

        lastPC = Inst.pc;
    }
    
    traindata.close();
    trace.close();
    return 0;
}