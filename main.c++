#include <iostream>
#include <vector>
#include "mathematical/LinearAlgebra.hpp"

using namespace std;

vector<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};

int main(){
    LinearAlgebra la;
    la.file();
    cout<<"\n";
    int myNum[10] = {1,2,4};
    for(int i = 0 ; i<=4 ; i++){
        cout<<cars[i]<<"\n";    
    }

    return 0;
}