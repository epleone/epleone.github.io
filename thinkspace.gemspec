Gem::Specification.new do |spec|
	spec.name          = "thinkspace"
	spec.version       = "2.5.0"
	spec.authors       = ["epleone"]
	spec.email         = ["epleone@sina.com"]

	spec.summary       = "A minimalist Jekyll theme"
	spec.homepage      = "https://epleone.github.io/"
	spec.license       = "MIT"

	spec.metadata["plugin_type"] = "theme"

	spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|(LICENSE|README)((\.(txt|md|markdown)|$)))!i) }

	spec.add_runtime_dependency "jekyll", "~> 4.0.0"

	#spec.add_development_dependency "bundler", "~> 2.1.0"
	spec.add_development_dependency "rake", "~> 12.0"
end

