#!/usr/bin/perl

use v5.010;

my $n = @ARGV[0];
my $range = @ARGV[1];
$n = 32 if !$n;
$range = 100 if !$range;

open OUT, ">input.txt";
print OUT "$n\n";
print OUT int(rand($range));
for ($i = 1; $i < $n; $i++) {
    print OUT " " . int(rand($range));
}
print OUT "\n";
close OUT;
