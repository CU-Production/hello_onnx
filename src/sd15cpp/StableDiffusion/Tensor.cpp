#include "Tensor.h"

#define _USE_MATH_DEFINES
#include <math.h>


namespace MachineLearning
{
  const Ort::MemoryInfo Tensor::_ortMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Tensor::Tensor() :
    Type(TensorType::Unknown),
    Shape({ 0, 0, 0, 0 })
  { }

  Tensor::Tensor(TensorType type, size_t x, size_t y, size_t z, size_t w) :
    Type(type),
    Shape({ x, y, z, w })
  {
    AllocateBuffer();
  }

  Tensor::Tensor(TensorType type, shape_t shape) :
    Type(type),
    Shape(shape)
  {
    AllocateBuffer();
  }

  Tensor::Tensor(Tensor&& other) noexcept
  {
    *this = std::move(other);
  }

  Tensor& Tensor::operator=(Tensor&& other) noexcept
  {
    Buffer = move(other.Buffer);
    Type = other.Type;
    Shape = other.Shape;

    other.Reset();

    return *this;
  }

  void Tensor::Reset() noexcept
  {
    Type = TensorType::Unknown;
    Shape = { 0, 0, 0, 0 };
    Buffer.clear();
  }

  void Tensor::AllocateBuffer()
  {
    Buffer.resize(ByteCount());
  }

  size_t Tensor::ByteCount() const
  {
    size_t result = Shape[0] > 0 ? GetElementSize(Type) : 0;

    for (auto dimension : Shape)
    {
      if (dimension != 0) result *= dimension;
    }

    return result;
  }

  bool Tensor::IsValid() const
  {
    return Type != TensorType::Unknown && !Buffer.empty() && Buffer.size() == ByteCount();
  }

  void Tensor::ThrowIfInvalid() const
  {
    if (IsValid()) throw std::runtime_error("The tensor is invalid.");
  }

  Tensor::operator bool() const
  {
    return IsValid();
  }

  size_t Tensor::Size(size_t index) const
  {
    size_t result = 1;

    for (auto i = index; i < Shape.size(); i++)
    {
      if (Shape[i] > 0) result *= Shape[i];
    }

    return result;
  }

  Tensor Tensor::CreateRandom(shape_t shape, std::span<std::minstd_rand> randoms, float scale)
  {
    static std::uniform_real_distribution<float> floatDistribution(0.f, 1.f);

    Tensor result{ TensorType::Single, shape };

    for (size_t i = 0; i < shape[0]; i++)
    {
      for (auto& value : result.AsSubSpan<float>(i))
      {
        auto u1 = floatDistribution(randoms[i]);
        auto u2 = floatDistribution(randoms[i]);
        auto radius = sqrt(-2.f * log(u1));
        auto theta = 2.f * M_PI * u2;
        auto standardNormalRand = radius * cos(theta);

        value = standardNormalRand * scale;
      }
    }

    return result;
  }

  Tensor Tensor::Concat(const Tensor& other) const
  {
    if (!AreShapesEqual(Shape, other.Shape, 1)) throw std::invalid_argument("other");
    if (Type != other.Type) throw std::invalid_argument("other");

    auto shape = Shape;
    shape[0] += other.Shape[0];

    Tensor tensor{ Type, shape };

    auto baseByteCount = ByteCount();

    auto pTarget = tensor.AsPointer();
    memcpy(pTarget, AsPointer(), baseByteCount);
    memcpy(pTarget + baseByteCount, other.AsPointer(), other.ByteCount());

    return tensor;
  }

  bool Tensor::operator==(const Tensor& other) const
  {
    return Type == other.Type && Shape == other.Shape &&
      0 == memcmp(AsPointer(), other.AsPointer(), ByteCount());
  }

  bool Tensor::operator!=(const Tensor& other) const
  {
    return !(*this == other);
  }

//  Tensor Tensor::ToSingle() const
//  {
//    if (Type != TensorType::Half) throw std::bad_cast();
//
//    auto size = Size();
//
//    Tensor result{ TensorType::Single, Shape };
//    XMConvertHalfToFloatStream(reinterpret_cast<float*>(result.Buffer.data()), 4, reinterpret_cast<const HALF*>(Buffer.data()), 2, size);
//
//    return result;
//  }
//
//  Tensor Tensor::ToHalf() const
//  {
//    if (Type != TensorType::Single) throw std::bad_cast();
//
//    auto size = Size();
//
//    Tensor result{ TensorType::Half, Shape };
//    XMConvertFloatToHalfStream(reinterpret_cast<HALF*>(result.Buffer.data()), 2, reinterpret_cast<const float*>(Buffer.data()), 4, size);
//
//    return result;
//  }

  size_t Tensor::GetDimensionFromIndex(size_t& x, size_t& y, size_t& z, size_t& w)
  {
    auto dimension = 4;

    if (w == ~0u)
    {
      dimension = 3;
      w = 0;
    }

    if (z == ~0u)
    {
      dimension = 2;
      z = 0;
    }

    if (y == ~0u)
    {
      dimension = 1;
      y = 0;
    }

    if (x == ~0u)
    {
      dimension = 0;
      x = 0;
    }

    return dimension;
  }

  bool Tensor::AreShapesEqual(shape_t a, shape_t b, size_t startDimension)
  {
    for (size_t i = startDimension; i < shape_dimension; i++)
    {
      if (a[i] != b[i]) return false;
    }

    return true;
  }

  size_t Tensor::ElementCount(shape_t shape)
  {
    size_t result = shape[0] > 0 ? 1 : 0;
    for (auto item : shape)
    {
      if (item) result *= item;
      else break;
    }
    return result;
  }

  std::vector<std::vector<float>> Tensor::ToTextureDataRgba8(ColorNormalization normalization) const
  {
    std::vector<std::vector<float>> results;
    results.reserve(Shape[0]);

    auto width = uint32_t(Shape[3]);
    auto height = uint32_t(Shape[2]);
    for (size_t i = 0u; i < Shape[0]; i++)
    {
      std::vector<float> result;
      results.reserve(width * height * 4);

      auto rSource = AsPointer<float>(i, 0);
      auto gSource = AsPointer<float>(i, 1);
      auto bSource = AsPointer<float>(i, 2);
      for (uint32_t y = 0u; y < height; y++)
      {
        for (uint32_t x = 0u; x < width; x++)
        {
          std::array<float, 4> color{ *bSource++, *gSource++, *rSource++, 1.f };
          switch (normalization)
          {
          case ColorNormalization::LinearPlusMinusOne:
            color = { color[0] / 2.f + 0.5f, color[1] / 2.f + 0.5f, color[2] / 2.f + 0.5f, 1.f };
            break;
          case ColorNormalization::LinearZeroToOne:
            color = { color[0], color[1], color[2], 1.f };
            break;
          default:
            throw std::logic_error("Normalization mode not implemented.");
          }
          result.push_back(color[0]);
          result.push_back(color[1]);
          result.push_back(color[2]);
          result.push_back(color[3]);
        }
      }

      results.push_back(std::move(result));
    }

    return results;
  }

  std::vector<std::vector<float>> Tensor::ToTextureData(ColorNormalization normalization) const
  {
    if (Type != TensorType::Single) throw std::bad_cast();

    switch (Shape[1])
    {
//    case 1:
//      return ToTextureDataGray8(normalization);
    case 3:
      return ToTextureDataRgba8(normalization);
    default:
      throw std::logic_error("Unsupported channel count.");
    }
  }

  void Tensor::Clamp(float min, float max)
  {
    if (Type != TensorType::Single) throw std::bad_cast();
    
    auto dataPtr = AsPointer<float>();
    size_t totalSize = Size();
    
    for (size_t i = 0; i < totalSize; i++)
    {
      dataPtr[i] = std::clamp(dataPtr[i], min, max);
    }
  }

  std::vector<std::vector<uint8_t>> Tensor::ToTextureRGBA8DataEx(ColorNormalization normalization) const
  {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(Shape[0]);
    auto width = uint32_t(Shape[3]);
    auto height = uint32_t(Shape[2]);
    for (size_t i = 0u; i < Shape[0]; i++)
    {
      std::vector<uint8_t> result;
      results.reserve(width * height * 4);
      auto rSource = AsPointer<float>(i, 0);
      auto gSource = AsPointer<float>(i, 1);
      auto bSource = AsPointer<float>(i, 2);
      for (uint32_t y = 0u; y < height; y++)
      {
        for (uint32_t x = 0u; x < width; x++)
        {
          std::array<float , 4> color{ *rSource++, *gSource++, *bSource++, 1.f };
          std::array<uint8_t , 4> byteColor{ 0, 0, 0, 255 };
          switch (normalization)
          {
            case ColorNormalization::LinearPlusMinusOne:
              byteColor = {
                (uint8_t)std::round(std::clamp(color[0] / 2.f + 0.5f, 0.0f, 1.0f) * 255),
                (uint8_t)std::round(std::clamp(color[1] / 2.f + 0.5f, 0.0f, 1.0f) * 255),
                (uint8_t)std::round(std::clamp(color[2] / 2.f + 0.5f, 0.0f, 1.0f) * 255),
                255 };
              break;
            case ColorNormalization::LinearZeroToOne:
              byteColor = {
                (uint8_t)std::round(std::clamp(color[0], 0.0f, 1.0f) * 255),
                (uint8_t)std::round(std::clamp(color[1], 0.0f, 1.0f) * 255),
                (uint8_t)std::round(std::clamp(color[2], 0.0f, 1.0f) * 255),
                255 };
              break;
            default:
              throw std::logic_error("Normalization mode not implemented.");
          }
          result.push_back(byteColor[0]);
          result.push_back(byteColor[1]);
          result.push_back(byteColor[2]);
          result.push_back(byteColor[3]);
        }
      }
      results.push_back(std::move(result));
    }
    return results;
  }

  const uint8_t* Tensor::AsPointer(size_t x, size_t y, size_t z, size_t w) const
  {
    shape_t index{ x, y, z, w };

    auto elementSize = GetElementSize(Type);
    size_t offset = 0;
    for (size_t i = 0; i < Shape.size(); i++)
    {
      offset += index[i] * Size(i + 1) * elementSize;
    }

    if (offset > Buffer.size()) throw std::out_of_range("Tensor index out of range.");

    return Buffer.data() + offset;
  }

  uint8_t* Tensor::AsPointer(size_t x, size_t y, size_t z, size_t w)
  {
    return const_cast<uint8_t*>(static_cast<const Tensor*>(this)->AsPointer(x, y, z, w));
  }

  Tensor Tensor::DuplicateToSize(size_t instances) const
  {
    if (Shape[0] == instances) return *this;

    if (instances % Shape[0] != 0) throw std::out_of_range("The instance count must be a multiple of the current shape[0].");

    return Duplicate(instances / Shape[0]);
  }

  Tensor Tensor::Duplicate(size_t instances) const
  {
    Tensor tensor{ Type, Shape[0] * instances, Shape[1], Shape[2], Shape[3] };

    for (size_t i = 0; i < instances; i++)
    {
      memcpy(tensor.AsPointer(i * Shape[0]), AsPointer(), ByteCount());
    }

    return tensor;
  }

  Tensor Tensor::Swizzle(size_t blockCount) const
  {
    Tensor result{ Type, Shape };

    auto blockByteCount = ByteCount() / Shape[0];
    auto blockSize = Shape[0] / blockCount;
    for (size_t i = 0; i < blockCount; i++)
    {
      for (size_t j = 0; j < blockSize; j++)
      {
        memcpy(result.AsPointer(i * blockSize + j), AsPointer(j * blockCount + i), blockByteCount);
      }
    }

    return result;
  }

  Tensor Tensor::Reshape(shape_t shape) const
  {
    if (ElementCount(shape) != ElementCount(Shape)) throw std::bad_cast();

    auto result{ *this };
    result.Shape = shape;
    return result;
  }

  std::vector<Tensor> Tensor::Split(size_t instances) const
  {
    if (Shape[0] % instances != 0) throw std::invalid_argument("instances");

    auto newShape = Shape;
    newShape[0] /= instances;

    std::vector<Tensor> results;
    results.resize(instances);
    for (size_t i = 0; auto & result : results)
    {
      result = Tensor(Type, newShape);
      memcpy(result.AsPointer(), AsPointer(i), result.ByteCount());

      i += newShape[0];
    }

    return results;
  }

  Tensor Tensor::FromOrtValue(const Ort::Value& value)
  {
    Tensor result;

    //Set type and shape
    auto info = ToTypeAndShape(value.GetTensorTypeAndShapeInfo());
    result.Type = info.first;
    result.Shape = info.second;
    result.AllocateBuffer();

    //Copy data
    memcpy(result.Buffer.data(), value.GetTensorRawData(), result.Buffer.size());

    return result;
  }

  Ort::Value Tensor::ToOrtValue() const
  {
    std::vector<int64_t> shape;
    for (auto dimension : Shape)
    {
      if (dimension != 0) shape.push_back(int64_t(dimension));
    }

    return Ort::Value::CreateTensor(_ortMemoryInfo, const_cast<uint8_t*>(Buffer.data()), Buffer.size(), shape.data(), shape.size(), ToTensorType(Type));
  }

  void Tensor::UpdateOrtValue(Ort::Value& value)
  {
    auto info = ToTypeAndShape(value.GetTensorTypeAndShapeInfo());
    if (info.first != Type || info.second != Shape) throw std::bad_cast();

    auto data = value.GetTensorMutableRawData();
    memcpy(data, AsPointer(), ByteCount());
  }

  std::pair<TensorType, Tensor::shape_t> Tensor::ToTypeAndShape(const Ort::TensorTypeAndShapeInfo& info)
  {
    std::pair<TensorType, Tensor::shape_t> result;

    //Convert type
    result.first = ToTensorType(info.GetElementType());

    //Convert shape
    auto shape = info.GetShape();
    if (shape.size() > result.second.size()) throw std::logic_error("Tensor does not support more than 4 dimensions.");

    for (auto i = 0; auto dimension : shape)
    {
      if (dimension > 0) result.second[i++] = size_t(dimension);
    }

    return result;
  }
}