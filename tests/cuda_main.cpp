
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
        printf("./cuda_example [test_name] [flag: 1 means verify results]\n");
        return 0;
    }

    if (argc > 2)
    {
        auto name = argv[1];
        int  flag = atoi(argv[2]);
        if (strcmp(name, "all") == 0)
        {
            return TestSuite::runAll();
        }
        else
        {
            return TestSuite::run(name, flag);
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
