AVPROGS-$(CONFIG_FFMPEG)   += ffmpeg
AVPROGS-$(CONFIG_FFPLAY)   += ffplay
AVPROGS-$(CONFIG_FFPROBE)  += ffprobe
AVPROGS-$(CONFIG_FFHWINFO) += ffhwinfo

AVPROGS     := $(AVPROGS-yes:%=%$(PROGSSUF)$(EXESUF))
PROGS       += $(AVPROGS)

AVBASENAMES  = ffmpeg ffplay ffprobe ffhwinfo
ALLAVPROGS   = $(AVBASENAMES:%=%$(PROGSSUF)$(EXESUF))
ALLAVPROGS_G = $(AVBASENAMES:%=%$(PROGSSUF)_g$(EXESUF))

OBJS-ffmpeg +=                  \
    fftools/ffmpeg_demux.o      \
    fftools/ffmpeg_filter.o     \
    fftools/ffmpeg_hw.o         \
    fftools/ffmpeg_mux.o        \
    fftools/ffmpeg_mux_init.o   \
    fftools/ffmpeg_opt.o        \
    fftools/objpool.o           \
    fftools/sync_queue.o        \
    fftools/thread_queue.o      \

OBJS-ffhwinfo +=                \
    fftools/ffhwinfo_gpu.o      \
    fftools/ffhwinfo_gpu_drm.o  \
    fftools/ffhwinfo_gpu_dxgi.o \
    fftools/ffhwinfo_gpu_cuda.o \
    fftools/ffhwinfo_gpu_qsv.o  \
    fftools/ffhwinfo_gpu_amf.o  \
    fftools/ffhwinfo_utils.o

define DOFFTOOL
OBJS-$(1) += fftools/cmdutils.o fftools/opt_common.o fftools/$(1).o $(OBJS-$(1)-yes)
ifdef HAVE_GNU_WINDRES
OBJS-$(1) += fftools/fftoolsres.o
endif
$(1)$(PROGSSUF)_g$(EXESUF): $$(OBJS-$(1))
$$(OBJS-$(1)): | fftools
$$(OBJS-$(1)): CFLAGS  += $(CFLAGS-$(1))
$(1)$(PROGSSUF)_g$(EXESUF): LDFLAGS += $(LDFLAGS-$(1))
$(1)$(PROGSSUF)_g$(EXESUF): FF_EXTRALIBS += $(EXTRALIBS-$(1))
-include $$(OBJS-$(1):.o=.d)
endef

$(foreach P,$(AVPROGS-yes),$(eval $(call DOFFTOOL,$(P))))

all: $(AVPROGS)

fftools/ffprobe.o fftools/cmdutils.o: libavutil/ffversion.h | fftools
OUTDIRS += fftools

ifdef AVPROGS
install: install-progs install-data
endif

install-progs-yes:
install-progs-$(CONFIG_SHARED): install-libs

install-progs: install-progs-yes $(AVPROGS)
	$(Q)mkdir -p "$(BINDIR)"
	$(INSTALL) -c -m 755 $(AVPROGS) "$(BINDIR)"

uninstall: uninstall-progs

uninstall-progs:
	$(RM) $(addprefix "$(BINDIR)/", $(ALLAVPROGS))

clean::
	$(RM) $(ALLAVPROGS) $(ALLAVPROGS_G) $(CLEANSUFFIXES:%=fftools/%)
