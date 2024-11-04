#include<vector>
#include<iostream>
#include<sstream>
#include <cstdint>
#include <cassert>
using namespace std;

typedef uint64_t Tick;
class Instruction{
public:
    Tick fetchTick;
    Tick execTick;
    Tick commitTick;

    int opCode;

    int NumFlags = 39;
    vector<int> flags;

    int numSrcRegs, numDestRegs;
    vector<pair<int, int>> regVector;

    int isMem = 0;
    uint64_t memAddr = 0;

    int isBranch = 0;
    uint64_t pc;
public:
    Instruction():flags(NumFlags, 0){}
    ~Instruction(){}
    void read(string& s){
        stringstream ss(s);

        ss >> fetchTick >> execTick >> commitTick;

        ss >> opCode;

        for(int i = 0; i < NumFlags; i++){
            ss >> flags[i];
        }

        ss >> numSrcRegs;
        for(int i = 0; i < numSrcRegs; i++){
            int regClass, regIdx;
            ss >> regClass >> regIdx;
            regVector.emplace_back(regClass, regIdx);
        }

        ss >> numDestRegs;
        for(int i = 0; i < numDestRegs; i++){
            int regClass, regIdx;
            ss >> regClass >> regIdx;
            regVector.emplace_back(regClass, regIdx);
        }

        
        ss >> isMem >> memAddr;
        ss >> isBranch >> pc;
    }  
};
