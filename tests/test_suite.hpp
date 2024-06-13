#pragma once

#include "timer.hpp"
#include "test_utils.hpp"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

/** test case */
class TestCase
{
    friend class TestSuite;

public:
    /**
     * @brief deinitializer
     */
    virtual ~TestCase() = default;

    virtual bool run(std::vector<int> flags) = 0;

private:
    /** case name */
    std::string name;
};

/** test suite */
class TestSuite
{
public:
    /**
     * @brief deinitializer
     */
    ~TestSuite();
    /**
     * @brief get shared instance
     * @return shared instance
     */
    static TestSuite *get();

public:
    /**
     * @brief register runable test case
     * @param test test case
     * @param name case name
     */
    void add(TestCase *test, const char *name);

    static int runAll(std::vector<int> flags = {0});

    static int run(const char *name, std::vector<int> flags = {0});

private:
    /** get shared instance */
    static TestSuite *g_instance;
    /** registered test cases */
    std::vector<TestCase *> m_tests;
};

/**
 static register for test case
 */
template <class Case>
class TestRegister
{
public:
    /**
     * @brief initializer. register test case to suite.
     * @param name test case name
     */
    TestRegister(const char *name)
    {
        TestSuite::get()->add(new Case, name);
    }
    /**
     * @brief deinitializer
     */
    ~TestRegister()
    {
    }
};

#define TestSuiteRegister(Case, name) static TestRegister<Case> __r##Case(name)
