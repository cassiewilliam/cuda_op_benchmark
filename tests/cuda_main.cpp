
#include "test_suite.hpp"
#include "test_utils.hpp"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "-h") == 0)
    {
        printf("./CUDABench [test_name] [0:verify or 1:profiler] [param1] [param2]\n");
        return 0;
    }

    if (argc > 2)
    {
        auto name = argv[1];
        std::vector<int> flags;
        for (int i = 2; i < argc; ++i)
        {
            flags.push_back(atoi(argv[i]));
        }
        if (strcmp(name, "all") == 0)
        {
            return TestSuite::runAll();
        }
        else
        {
            return TestSuite::run(name, flags);
        }
    }
    else if (argc > 1)
    {
        auto name = argv[1];
        if (strcmp(name, "all") == 0)
        {
            return TestSuite::runAll();
        }
        else
        {
            return TestSuite::run(name);
        }
    }
    else
    {
        return TestSuite::runAll();
    }

    return 0;
}
