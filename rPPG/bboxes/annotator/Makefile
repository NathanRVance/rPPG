CC=g++
LIBS=libconfuse opencv4
CFLAGS=-c -Wall -fPIC -std=c++20 -Wno-deprecated-enum-enum-conversion
LDFLAGS=
SOURCES=src/annotator.cc src/labeller.cc src/playback.cc src/settings.cc src/ui.cc 
OBJECTS=$(SOURCES:.cc=.o)
LIBRARY=
EXECUTABLE=annotator
ifeq ($(PREFIX),)
    PREFIX := /usr
endif

all: $(SOURCES) $(EXECUTABLE)

install: $(EXECUTABLE)
	install -d $(DESTDIR)$(PREFIX)/bin/
	install -m 755 $(EXECUTABLE) $(DESTDIR)$(PREFIX)/bin/

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) `pkg-config $(LIBS) --libs`


src/annotator.o: src/annotator.cc src/playback.h src/ui.h src/labeller.h
	$(CC) $(CFLAGS) $< -o $@ `pkg-config $(LIBS) --cflags`

src/labeller.o: src/labeller.cc src/labeller.h src/settings.h
	$(CC) $(CFLAGS) $< -o $@ `pkg-config $(LIBS) --cflags`

src/playback.o: src/playback.cc src/playback.h
	$(CC) $(CFLAGS) $< -o $@ `pkg-config $(LIBS) --cflags`

src/settings.o: src/settings.cc src/settings.h
	$(CC) $(CFLAGS) $< -o $@ `pkg-config $(LIBS) --cflags`

src/ui.o: src/ui.cc src/ui.h src/playback.h src/labeller.h
	$(CC) $(CFLAGS) $< -o $@ `pkg-config $(LIBS) --cflags`

clean:
	rm -f src/*.o $(LIBRARY) $(EXECUTABLE)
