#pragma once

#include "enum.hpp"

#include <map>
#include <mutex>

class Operator;
class Creator
{
public:
    virtual Operator *onCreate(KernelType type) const = 0;
};

// NOTE: 用于Creator管理
class CreatorFactory
{
public:
    static bool addCreator(OpType t, Creator *c);
    static std::map<OpType, Creator *> *gCreator();
};

// NOTE: 用于Creator注册
template <class T>
class CreatorRegister
{
public:
    CreatorRegister(OpType type)
    {
        T *t = new T;
        CreatorFactory::addCreator(type, t);
    }
    ~CreatorRegister() = default;
};