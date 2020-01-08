# Compile and run every configuration of a simplified bug.
# 
# Note that this script just extracts the names of the configuration options
# from the .c file and tries every combination. So with n options, it will
# generate, compile, and run 2^n programs. Since the #define statements in
# the .c files ensure that the feature model is respected, this may lead to
# some redundant configurations.
#
# Also note that not all of the simplified bugs produce a compile or runtime
# error. Check the corresponding .bug report.


# set the name of the generated executable file
$out = "temp.out"

# set the C99 compiler command
$cc = "gcc -std=c99 -w -o #{$out}"

# get the names of all configuration options from a file
def configs(file)
  name  = /(?<macro>[A-Z0-9_]+)/
  regex = Regexp.union(/#ifdef #{name}/, /#(el)?if defined\(#{name}\)/)
  opts  = File.open(file).read.scan(regex).flatten.compact.uniq
  [[]] + (1 .. opts.length).map { |i| opts.combination(i).to_a }.flatten(1)
end

# the options to pass to $cc for a configuration
def config_opts(config)
  if config == []
    ""
  else
    "-D" + config.join(" -D")
  end
end

# test a particular configuration
def run(config, file)
  puts "\nConfiguration: #{config.join(" ")}"
  `#{$cc} #{config_opts(config)} #{file}`
  if $?.success?
    system(File.expand_path($out))
    if $?.success?
      puts "[success]"
    else
      puts "[runtime error]"
    end
  else
    puts "[compile error]"
  end
  File.delete($out) if File.exists?($out)
end

# test all configurations
ARGV.map do |f|
  puts "\nFile: #{f}" if ARGV.length > 1
  configs(f).map { |c| run(c,f) }
end
