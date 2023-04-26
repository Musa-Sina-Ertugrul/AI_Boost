%module AIBoost

%include <std_vector.i>
%include "std_vector.i"
%include "std_string.i"



%{
    #include "/home/musasina/Desktop/AI_Boost/AiLib/DataSet.hpp"
    #include "/home/musasina/Desktop/AI_Boost/AiLib/Enums.hpp"
    #include "/home/musasina/Desktop/AI_Boost/AiLib/Layer.hpp"
    #include "/home/musasina/Desktop/AI_Boost/AiLib/Model.hpp"
    #include <vector>
    #include <iostream>
    #include <Python.h>
    #include <vector>
    #include <string>
    #include <vector>
    #include <algorithm>
    #include <numeric>

%}


%include <std_vector.i>

%template(FloatVector) std::vector<double>;
%template(FloatVectorVector) std::vector<std::vector<double>>;
%template(LayerPtrVector) std::vector<Layer*>;
%{
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
%}



%include "/home/musasina/Desktop/AI_Boost/AiLib/DataSet.hpp"
%include "/home/musasina/Desktop/AI_Boost/AiLib/Enums.hpp"
%include "/home/musasina/Desktop/AI_Boost/AiLib/Layer.hpp"
%include "/home/musasina/Desktop/AI_Boost/AiLib/Model.hpp"
