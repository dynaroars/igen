obj-y := fork.o
ifeq ($(CONFIG_A),y)
  BITS := 32
else
  BITS := 64
endif
obj-$(CONFIG_B) += probe_$(BITS).o

c-flags-$(CONFIG_A) += -DCONFIG_A
c-flags-$(CONFIG_B) += -DCONFIG_B

built-in.o: $(obj-y)
	gcc -o $@ $^

%.o: %.c
	gcc -c $(c-flags-y) $^

clean:
	rm -f *.o
