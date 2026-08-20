node Person {
    has name: str;
    has age: int = 0;
}

walker Greeter {
    can greet with Person entry {
        print(f"Hello, {here.name}!");
    }
}

with entry {
    p = Person(name="Ada", age=36);
    root ++> p;
    p spawn Greeter();
}
