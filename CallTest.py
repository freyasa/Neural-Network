class Person:
    def __call__(self, name):
        print("__call__" + " Hello " + name)

    def hello(self, name):
        print("Hello " + name)


if __name__ == '__main__':
    person = Person()
    person("Kototo")
    person.hello("Kototo")