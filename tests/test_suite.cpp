#include "test_suite.hpp"
#include <stdlib.h>
#include <mutex>

TestSuite *TestSuite::g_instance = nullptr;

TestSuite *TestSuite::get()
{
    static std::once_flag s_once_flag;
    std::call_once(s_once_flag, [&]()
                   { g_instance = new (std::nothrow) TestSuite; });
    return g_instance;
}

TestSuite::~TestSuite()
{
    for (int i = 0; i < m_tests.size(); ++i)
    {
        delete m_tests[i];
    }
    m_tests.clear();
}

void TestSuite::add(TestCase *test, const char *name)
{
    test->name = name;
    m_tests.push_back(test);
}

static void printTestResult(int wrong, int right)
{
    printf("{\"failed\":%d,\"passed\":%d}\n", wrong, right);
}

int TestSuite::run(const char *key, std::vector<int> flags)
{
    if (key == NULL || strlen(key) == 0)
        return 0;

    auto suite = TestSuite::get();
    std::string prefix = key;
    std::vector<std::string> wrongs;
    size_t runUnit = 0;
    for (int i = 0; i < suite->m_tests.size(); ++i)
    {
        TestCase *test = suite->m_tests[i];
        if (test->name.find(prefix) == 0)
        {
            runUnit++;
            printf("\trunning %s.\n", test->name.c_str());
            auto res = test->run(flags);
            if (!res)
            {
                wrongs.emplace_back(test->name);
            }
        }
    }
    if (wrongs.empty())
    {
        printf("√√√ all <%s> tests passed.\n", key);
    }
    for (auto &wrong : wrongs)
    {
        printf("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), runUnit - wrongs.size());
    return wrongs.size();
}

int TestSuite::runAll(std::vector<int> flags)
{
    auto suite = TestSuite::get();
    std::vector<std::string> wrongs;
    for (int i = 0; i < suite->m_tests.size(); ++i)
    {
        TestCase *test = suite->m_tests[i];
        printf("\trunning %s.\n", test->name.c_str());
        auto res = test->run(flags);
        if (!res)
        {
            wrongs.emplace_back(test->name);
        }
    }
    if (wrongs.empty())
    {
        printf("√√√ all tests passed.\n");
    }
    for (auto &wrong : wrongs)
    {
        printf("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), suite->m_tests.size() - wrongs.size());
    return wrongs.size();
}
