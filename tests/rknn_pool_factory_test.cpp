#include "rknnPool.hpp"

#include <cassert>
#include <memory>
#include <string>

namespace {

class FakeModel
{
public:
    FakeModel(const std::string& model_path, int offset)
        : model_path_(model_path), offset_(offset), context_(0)
    {
    }

    int init(int*, bool)
    {
        return model_path_ == "model.rknn" ? 0 : -1;
    }

    int* get_pctx()
    {
        return &context_;
    }

    int infer(int input)
    {
        return input + offset_;
    }

private:
    std::string model_path_;
    int offset_;
    int context_;
};

}  // namespace

int main()
{
    int created_models = 0;
    rknnPool<FakeModel, int, int> pool(
        "model.rknn",
        2,
        [&created_models](const std::string& model_path) {
            ++created_models;
            return std::make_shared<FakeModel>(model_path, 5);
        });

    assert(pool.init() == 0);
    assert(created_models == 2);
    assert(pool.put(7) == 0);

    int result = 0;
    assert(pool.get(result) == 0);
    assert(result == 12);
}
