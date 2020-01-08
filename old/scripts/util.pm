package util;

use strict;
use warnings;
use Data::Dumper;


sub readConfigurationSpace($){
    my $file = shift;

    open(IN, "< $file")
	|| die("Cannot open $file\n");

    # ignore config count
    my $cfgCnt = <IN>;
    chomp($cfgCnt);

    my %cfgToTest=();

    my $title = <IN>;
    chomp($title);
    $title=~s /\,testList$//g;
    my @title = split(/,/, $title);


    my @cfgs = ();
    while(<IN>){
	#chomp;

	my $ln = $_;

	$ln =~ s/^\s+//g;
	$ln =~ s/\s+$//g;
	next if ($ln eq "");

	my $tests=undef;
	if ($ln=~m /\{((.|\[|\])+)\}$/ ){
	    $tests=$1;

	}
	else{
	    print "WARNING: UNRECOGNIZED LINE $ln\n";
	    next;
	}

	#$ln=~s /"\$tests"//g;

	#$ln=~s /\{//g;
	#$ln=~s /\}//g;
	#$ln=~s /\[//g;
	#$ln=~s /\]//g;
	my $space=index($ln," ");

	$ln=substr($ln,0,$space);

	$ln =~ s/\s+$//g;



	my @cfg = split(/,/, $ln);
	my @test = split(/,/, $tests);


	my $cfgStr=join(" ",@cfg);

	$cfgToTest{$cfgStr}=\@test;

	push(@cfgs, \@cfg);
    }

    close(IN);

    my %cfgSpace = ();
    $cfgSpace{opts} = \@title;
    $cfgSpace{cfgs} = \@cfgs;
    $cfgSpace{cfgToTest}=\%cfgToTest;

    return \%cfgSpace;
}



sub readConfigurationModel($){
    my $fileName = shift;

    my @configModel = ();

    open(IN, "< $fileName")
	|| die("ERROR: Cannot open $fileName\n");
    while(<IN>){
	chomp;

	my $ln = $_;
	$ln =~ s/^\s+//g;
	$ln =~ s/\s+$//g;

	next if ($ln eq "");

	my $optValues = undef;
	my $optName = undef;
	my $optType = undef;

	if ($ln =~ m/^\s*([^\,]+)\s*\,\s*\{(.+)\}\s*\,\s*([^\,]+)\s*$/){
	    $optName = $1;
	    $optValues = $2;
	    $optType = $3;

	    $optName =~ s/^\s+//g;
	    $optName =~ s/\s+$//g;

	    $optType =~ s/^\s+//g;
	    $optType =~ s/\s+$//g;

	    my @optValues = split(/,/, $optValues);
	    for(my $idx = 0; $idx < scalar @optValues; $idx++){
		my $optValue = $optValues[$idx];
		$optValue =~ s/^\s+//g;
		$optValue =~ s/\s+$//g;
		$optValues[$idx] = $optValue;
	    }

	    my %info = ();
	    $info{name} = $optName;
	    $info{type} = $optType;
	    $info{settings} = \@optValues;

	    push(@configModel, \%info);

	}
	else{
	    print "WARNING: Something is wrong with the following line:\n$ln\n";
	    next;
	}
    }
    close(IN);

    #print Dumper(\@configModel);

    return \@configModel;
}

sub isCompileTimeOption($$){
    my $optionName = shift;
    my $cfgModel = shift;

    foreach my $opt (@{$cfgModel}){
	if ($opt->{name} eq $optionName){
	    return  1
		if ($opt->{type} eq "compile-time");
	    return 0;
	}
    }

    die("ERROR: Unknown option ($optionName)\n")
	if (1);
}


sub processLCovRecord($$){
    my $record = shift;
    my $harvest = shift;

    my $sourceFile = $record->{sourceFile};
    foreach my $elm (@{$record->{lst}}){
	if ($elm =~ m/^DA\:\s*([0-9]+)\,([0-9]+)\s*$/){
	    my $lineNo = $1;
	    my $executionCnt = $2;

	    next if ($executionCnt eq "0");

	    if (! exists $harvest->{$sourceFile}){
		my %tmp = ();
		$harvest->{$sourceFile} = \%tmp;
	    }
	    $harvest->{$sourceFile}->{$lineNo} = "";
	}
	else{
	    #print "WARNING: Something is worng with the following line\n\t$elm\n";
	    next;
	}

    }

}


sub parseLCovFile($$){
    my $fileName = shift;
    my $sutDir = shift;

    my %harvest = ();
    my $recording = 0;
    my %record = ();

    open(IN, "< $fileName")
	|| die("Cannot open $fileName\n");
    while(<IN>){
	chomp;

	my $ln = $_;
	$ln =~ s/^\s+//g;
	$ln =~ s/\s+$//g;

	if (($recording) && ($ln eq "end_of_record")){
	    processLCovRecord(\%record, \%harvest);
	    $recording = 0;
	    next;
	}

	if ((! $recording) && ($ln =~ m/^SF\:\s*(.+)\s*$/)){
	    my $sourceFile = $1;

	    next if ($sourceFile !~ m/^$sutDir\//);
	    $sourceFile =~ s/^$sutDir\///;

	    %record = ();
	    my @lst = ();
	    $record{sourceFile} = $sourceFile;
	    $record{lst} = \@lst;
	    $recording = 1;
	}

	if ( ($recording) && ($ln =~ m/^DA\:/) ){
	    push(@{$record{lst}}, $ln);
	}


    }

    close(IN);

    #print Dumper(\%harvest);

    return \%harvest;
}


sub parseConfigSpaceFileName($){
    my $configSpaceFileName = shift;

    my $baseCfgSpaceName = undef;
    my $roundNo = undef;
    my $stepNo = undef;
    my $path = undef;


    if ($configSpaceFileName =~ m/\.([0-9]+)\.s([0-9]+)$/){
	$roundNo = $1;
	$stepNo = $2;
    }
    elsif ($configSpaceFileName =~ m/\.s([0-9]+)$/){
	$stepNo = $1;
	$roundNo = 0;
    }
    elsif ($configSpaceFileName =~ m/\.([0-9]+)$/){
	$roundNo = $1;
	$stepNo = 0;
    }
    else{
	$roundNo = 0;
	$stepNo = 0;

    }

    $configSpaceFileName =~ s/\.s([0-9]+)$//g;
    $configSpaceFileName =~ s/\.([0-9]+)$//g;

    if ($configSpaceFileName =~ m/^((.+)\/)?([^\/]+)$/){
	$path = $2;
	$path = "." if (! defined $path);
	$baseCfgSpaceName = $3;
    }

    my %info = ();
    $info{baseCfgSpaceName} = $baseCfgSpaceName;
    $info{roundNo} = $roundNo;
    $info{stepNo} = $stepNo;
    $info{path} = $path;

    return \%info;
}

# compute lcovA set difference lcovB
sub lcovDifference($$){
    my $lcovA = shift;
    my $lcovB = shift;

    my %diff  = ();
    foreach my $file (keys %{$lcovA}){
	foreach my $lineNo (keys %{$lcovA->{$file}}){

	    next if (exists $lcovB->{$file}->{$lineNo});

	    if (! exists $diff{$file}){
		my %tmp = ();
		$diff{$file} = \%tmp;
	    }

	    $diff{$file}->{$lineNo} = "";
	}

    }

    return \%diff;
}



sub printLCovInfo($$){
    my $lcovInfo = shift;
    my $outFile = shift;

    open(OUT, "> $outFile")
	|| die("Cannot create $outFile\n");
    foreach my $sourceName (sort keys %{$lcovInfo}){
	foreach my $lineNo (sort keys %{$lcovInfo->{$sourceName}}){
	    print OUT "$sourceName,$lineNo\n";
	}
    }
    close(OUT);

}



sub harvestLCovFile($$$$){
    my $aggregatedCovFile = shift;
    my $configSpaceFile = shift;
    my $sutDir = shift;
    my $outFile = shift;

    my $lcovInfo = parseLCovFile($aggregatedCovFile, $sutDir);

    # Find the last aggregated coverage file
    # that is the one that exists before the current one
    # Note that aggragtedCovFile contains all the coverage information
    # that has been obtained so far
    # in order to be able to find the current coverage we need to take the difference
    if ($aggregatedCovFile =~ m/^(.+)\/aggregated\.(.+)\.cov$/){
	my $path = $1;
	my $configSpaceName = $2;

	my $configSpaceInfo =  parseConfigSpaceFileName($configSpaceName);
	my $baseConfigSpaceName = $configSpaceInfo->{baseCfgSpaceName};
	my $roundNo = $configSpaceInfo->{roundNo};
	my $stepNo = $configSpaceInfo->{stepNo};

	my $previousCovFile = undef;

	# take a look at the previous step
	for(my $step = 1; $step < $stepNo; $step++){
	    my $covFile = "$path/aggregated.$baseConfigSpaceName";
	    $covFile = "$covFile.$roundNo" if ($roundNo > 0);
	    $covFile = "$covFile.s$step";
	    $covFile = "$covFile.cov";
	    $previousCovFile = $covFile if (-e $covFile);
	}

	if (! defined $previousCovFile){
	    # if there is no previous step
	    # take a look at the previous round

	    # Note that at this point you should not take a look at the steps anymore
	    for(my $round = 0; $round < $roundNo; $round++){
		my $covFile = "$path/aggregated.$baseConfigSpaceName";
		$covFile = "$covFile.$round" if ($round > 0);
		$covFile = "$covFile.cov";
		$previousCovFile = $covFile if (-e $covFile);
	    }
	}

	my %tmp = ();
	my $previousLcovInfo = \%tmp;
	$previousLcovInfo = parseLCovFile($previousCovFile, $sutDir)
	    if (defined $previousCovFile);

	my $coverageDiff = lcovDifference($lcovInfo, $previousLcovInfo);

	printLCovInfo($coverageDiff, $outFile);
    }
    else{
	die("ERROR: Malformed file name: $aggregatedCovFile. Exiting...\n");
    }
}



1;
