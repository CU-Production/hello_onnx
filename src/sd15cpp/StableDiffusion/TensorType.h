#pragma once

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_float16.h>

namespace MachineLearning
{
  enum class TensorType
  {
    Unknown,
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Half,
    Single,
    Double
  };

  template<typename T>
  constexpr TensorType ToTensorType()
  {
    if constexpr (std::is_same_v<T, bool>)
    {
      return TensorType::Bool;
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
      return TensorType::UInt8;
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
      return TensorType::UInt16;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
      return TensorType::UInt32;
    }
    else if constexpr (std::is_same_v<T, uint64_t>)
    {
      return TensorType::UInt64;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
      return TensorType::Int8;
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
      return TensorType::Int16;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
      return TensorType::Int32;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
      return TensorType::Int64;
    }
    else if constexpr (std::is_same_v<T, Ort::Float16_t>)
    {
      return TensorType::Half;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
      return TensorType::Single;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
      return TensorType::Double;
    }
    else
    {
      return TensorType::Unknown;
    }
  }

  size_t GetElementSize(TensorType type);

  TensorType ToTensorType(ONNXTensorElementDataType type);
  ONNXTensorElementDataType ToTensorType(TensorType type);
}