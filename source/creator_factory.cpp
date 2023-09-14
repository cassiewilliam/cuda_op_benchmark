#include "creator_factory.hpp"

std::map<OpType, Creator *> *CreatorFactory::gCreator()
{
    static std::map<OpType, Creator *> *s_instance = nullptr;
    static std::once_flag s_once_flag;
    std::call_once(s_once_flag, [&]()
                   { s_instance = new (std::nothrow) std::map<OpType, Creator *>; });
    return s_instance;
};

bool CreatorFactory::addCreator(OpType t, Creator *c)
{
    auto map = gCreator();
    if (map->find(t) != map->end())
    {
        printf("Info: %d creator has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}