#include <cstddef> // for std::size_t

namespace ipc {
// A minimal, non-owning view of a contiguous sequence of objects.
template <typename T> class span {
public:
    // Member types
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;

    // Constructors
    // Default constructor (creates an empty span)
    constexpr span() noexcept : ptr_(nullptr), size_(0) { }

    // Construct from a pointer and a count
    constexpr span(pointer ptr, size_type count) noexcept
        : ptr_(ptr)
        , size_(count)
    {
    }

    // Construct from a pointer and an end pointer
    constexpr span(pointer first, pointer last) noexcept
        : ptr_(first)
        , size_(static_cast<size_type>(last - first))
    {
    }

    // Element access
    constexpr reference operator[](size_type idx) const noexcept
    {
        // In a real implementation, bounds checking might be optional (e.g., in
        // debug builds).
        return *(ptr_ + idx);
    }

    constexpr pointer data() const noexcept { return ptr_; }

    // Observers
    constexpr size_type size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }

    // Iterators
    constexpr iterator begin() const noexcept { return ptr_; }
    constexpr iterator end() const noexcept { return ptr_ + size_; }

private:
    pointer ptr_;
    size_type size_;
};
} // namespace ipc
