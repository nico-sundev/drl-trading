# Dependency Injection Framework Comparison

## Current: dependency-injector (Verbose Manual Wiring)

**Pros:**
- Explicit control over wiring
- Mature and well-documented

**Cons:**
- Extremely verbose (200+ lines for your container)
- Manual wiring of every dependency
- No type safety
- Hard to maintain and refactor
- Not intuitive for developers from Spring background

**Example:**
```python
preprocess_service = providers.Singleton(
    PreprocessService,
    features_config=features_config,
    feature_class_registry=feature_class_factory,
    feature_aggregator=feature_aggregator,
    merge_service=merge_service,
    context_feature_service=context_feature_service,
)
```

---

## Option 1: pinject (Google's Framework - Most Automatic)

**Pros:**
- ✅ **Zero configuration** - automatically wires based on parameter names
- ✅ **Closest to Spring's magic** - no decorators needed
- ✅ **Minimal boilerplate** - classes are automatically injectable
- ✅ **Convention over configuration**
- ✅ **Production-ready** (used by Google)

**Cons:**
- Less explicit than other options
- Parameter name matching can be fragile

**Example:**
```python
class PreprocessService:
    def __init__(self, features_config, feature_aggregator, merge_service):
        # pinject automatically finds and injects all dependencies!
        pass

obj_graph = pinject.new_object_graph(modules=[ConfigModule()])
service = obj_graph.provide(PreprocessService)  # Everything auto-wired!
```

---

## Option 2: injector (Type-Safe and Pythonic)

**Pros:**
- ✅ **Full type safety** with type hints
- ✅ **@inject decorators** similar to Spring's @Autowired
- ✅ **@singleton** decorators for lifecycle management
- ✅ **Provider methods** for complex configuration
- ✅ **IDE support** - full autocomplete and type checking
- ✅ **Explicit but concise**

**Cons:**
- Requires @inject decorators on constructors
- Slightly more verbose than pinject

**Example:**
```python
@singleton
class PreprocessService:
    @inject
    def __init__(
        self,
        feature_aggregator: FeatureAggregator,
        merge_service: MergeService
    ):
        pass

injector = Injector([ApplicationModule()])
service = injector.get(PreprocessService)  # Type-safe injection!
```

---

## Option 3: lagom (Spring-Like Containers)

**Pros:**
- ✅ **@injectable decorators** similar to Spring's @Component
- ✅ **Container-based** approach like Spring
- ✅ **Environment variable integration**
- ✅ **Familiar to Spring developers**

**Cons:**
- Less mature than other options
- More manual container configuration

**Example:**
```python
@injectable
class PreprocessService:
    def __init__(self, feature_aggregator: FeatureAggregator):
        pass

@container
class ApplicationContainer:
    # Auto-wired based on @injectable decorators
    pass

service = container.resolve(PreprocessService)
```

---

## Recommendation: **injector** (Best Balance)

For your trading system, I recommend **injector** because:

1. **Type Safety**: Full type checking with mypy integration
2. **Spring-Like**: `@inject` decorators feel familiar
3. **Production Ready**: Used in many production systems
4. **IDE Support**: Full autocomplete and refactoring support
5. **Explicit but Clean**: Clear what's being injected without verbosity
6. **Module System**: Clean separation of configuration
7. **Lifecycle Management**: Built-in singleton support

## Migration Strategy

1. **Phase 1**: Add injector to common library dependencies
2. **Phase 2**: Create new DI module alongside existing container
3. **Phase 3**: Migrate bootstrap classes to use injector
4. **Phase 4**: Gradually migrate services to use @inject
5. **Phase 5**: Remove old dependency-injector code

Your container would go from **200+ lines** to **~50 lines** of clean, type-safe code!

## Next Steps

Would you like me to:
1. Create a complete migration plan for your specific services?
2. Implement the new DI system in your common library?
3. Update your bootstrap classes to use the new system?
4. Show how to integrate with your existing message bus factory?
